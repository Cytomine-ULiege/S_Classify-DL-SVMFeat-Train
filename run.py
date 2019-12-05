import os
import torch
import numpy as np
from pathlib import Path
from mtdp import build_model
from cytomine import CytomineJob
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.utils import check_random_state
from torchvision.datasets import ImageFolder
from cytomine.models import AttachedFile, Job, Property
from torchvision.transforms import Resize, transforms
from cytomine.utilities.software import setup_classify, parse_domain_list, stringify
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader

NETWORKS = {'resnet50', 'densenet121'}


def parse_list_or_none(lst, cvt=int):
    return [] if lst is None else [cvt(s.strip()) for s in lst.split(",")]


def normCenterCropTransform(size):
    return transforms.Compose([
        Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(status=Job.RUNNING, statusComment="Initialization.")
        random_state = check_random_state(cj.parameters.seed)

        if cj.parameters.network not in NETWORKS:
            raise ValueError("Invalid value (='{}'} for parameter 'network'.".format(cj.parameters.network))
        if cj.parameters.classifier not in {"svm"}:
            raise ValueError("Invalid value (='{}') for parameter 'classifier'.".format(cj.parameters.classifier))
        if cj.parameters.pretrained not in {"imagenet", "mtdp"}:
            raise ValueError("Unknown value (='{}') for parameter 'pretrained'.".format(cj.parameters.pretrained))

        # prepare paths
        working_path = str(Path.home())
        in_path = os.path.join(working_path, "data")
        setup_classify(
            args=cj.arguments, logger=cj.logger,
            root_path=working_path, image_folder="data",
            dest_pattern=os.path.join("{term}", "{image}_{id}.png"),
            showTerm=True, showMeta=True, showWKT=True
        )

        cj.job.update(progress=15, statusComment="Prepare data loader....")
        width = 224
        dataset = ImageFolderWithPaths(in_path, transform=normCenterCropTransform(width))
        sampler = BatchSampler(SequentialSampler(dataset), batch_size=cj.parameters.batch_size, drop_last=False)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cj.parameters.n_jobs)

        cj.job.update(progress=17, statusComment="Load network...")
        device = torch.device("cpu")

        network = build_model(
            arch=cj.parameters.network,
            pretrained=cj.parameters.pretrained,
            pool=True
        )
        network.to(device)
        network.eval()

        n_samples = len(dataset)
        n_features = network.features.n_features()
        x = np.zeros([n_samples, n_features], dtype=np.float)
        y = np.zeros([n_samples], dtype=np.int)
        groups = np.zeros([n_samples], dtype=np.int)
        prev_start = 0

        for (x_batch, y_batch, p_batch) in cj.monitor(loader, start=20, end=60, period=0.025, prefix="Extract features"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            f_batch = network.forward(x_batch)
            end = prev_start + f_batch.shape[0]
            x[prev_start:end] = f_batch.detach().cpu().numpy()
            y[prev_start:end] = y_batch
            groups[prev_start:end] = [int(os.path.basename(p).split("_", 1)[0]) for p in p_batch]
            prev_start = end

        cj.logger.info("Transform classes...")
        classes = parse_domain_list(cj.parameters.cytomine_id_terms)
        positive_classes = parse_domain_list(cj.parameters.cytomine_positive_terms)
        classes = np.array(classes) if len(classes) > 0 else np.unique(y)
        n_classes = classes.shape[0]
        keep = np.in1d(y, classes)
        x, y = x[keep], y[keep]

        if cj.parameters.cytomine_binary:
            cj.logger.info("Will be training on 2 classes ({} classes before binarization).".format(n_classes))
            y = np.in1d(y, positive_classes).astype(np.int)
            n_classes = 2
        else:
            cj.logger.info("Will be training on {} classes.".format(n_classes))
            y = np.searchsorted(classes, y)

        cj.logger.info("Data:")
        cj.logger.info("> features: {}".format(x.shape))
        cj.logger.info("> unique classes: {}".format(np.unique(y).shape[0]))
        cj.logger.info("> unique labels : {}".format(np.unique(groups).shape[0]))

        cj.job.update(progress=60, prefix="Prepare cross validation...")
        unique_labels = np.unique(groups)
        n_splits = min(cj.parameters.cv_folds, unique_labels.shape[0])
        if n_splits == 1:  # if only one split, split on data instead of images
            cv = KFold(n_splits=min(cj.parameters.cv_folds, n_samples), shuffle=True, random_state=random_state)
        else:
            cv = GroupKFold(n_splits=n_splits)

        cj.job.update(progress=62, prefix="Prepare classifier '{}'...".format(cj.parameters.classifier))
        model = LinearSVC()

        grid_search = GridSearchCV(
            model, cv=cv, verbose=10, refit=True,
            param_grid={"C": [10 ** (-i) for i in range(-1, 11)]},
            scoring=make_scorer(accuracy_score) if n_classes > 2 else make_scorer(roc_auc_score, needs_threshold=True),
            n_jobs=cj.parameters.n_jobs if cj.parameters.classifier == "svm" else 1
        )

        cj.job.update(progress=65, statusComment="Grid searching...")
        grid_search.fit(x, y, groups=groups)

        print("Grid search:")
        print("> best params: {}".format(grid_search.best_params_))
        print("> best score : {}".format(grid_search.best_score_))

        cj.job.update(progress=95, statusComment="Save model...")
        model_path = os.path.join(working_path, "model.joblib")
        joblib.dump(grid_search.best_estimator_, model_path)

        AttachedFile(
            cj.job,
            domainIdent=cj.job.id,
            filename=model_path,
            domainClassName="be.cytomine.processing.Job"
        ).upload()

        Property(cj.job, key="classes", value=stringify(classes)).save()
        Property(cj.job, key="binary", value=cj.parameters.cytomine_binary).save()
        Property(cj.job, key="positive_classes", value=stringify(positive_classes)).save()

        cj.job.update(status=Job.SUCCESS, statusComment="Finished.", progress=100)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
