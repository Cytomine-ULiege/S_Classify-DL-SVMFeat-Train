import os
import torch
import numpy as np
from pathlib import Path
from cytomine import CytomineJob
from cytomine.models import AttachedFile, Job, Property
from cytomine.utilities.annotations import get_annotations
from sklearn.externals import joblib
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.utils import check_random_state


from sklearn.svm import LinearSVC
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader

from beheaded_networks import densenet201, resnet50
from dataset import ImageFolderWithPaths, normCenterCropTransform


NETWORKS = {
#    "densenet": (1920, densenet201),
    "resnet": (2048, resnet50)
}


def parse_list_or_none(lst, cvt=int):
    return [] if lst is None else [cvt(s.strip()) for s in lst.split(",")]


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(status=Job.RUNNING, statusComment="Initialization.")
        random_state = check_random_state(cj.parameters.random_seed)

        if cj.parameters.network not in NETWORKS:
            raise ValueError("Invalid value (='{}'} for parameter 'network'.".format(cj.parameters.network))
        if cj.parameters.classifier not in {"svm"}:
            raise ValueError("Invalid value (='{}') for parameter 'classifier'.".format(cj.parameters.classifier))

        # prepare paths
        working_path = str(Path.home())
        in_path = os.path.join(working_path, "data")
        os.makedirs(in_path, exist_ok=True)

        # load and dump annotations
        cj.job.update(statusComment="Download annotations...")
        projects = {cj.parameters.cytomine_id_project}.union(parse_list_or_none(cj.parameters.cytomine_id_projects))
        annotations = get_annotations(
            projects=list(projects),
            images=parse_list_or_none(cj.parameters.cytomine_id_images),
            terms=parse_list_or_none(cj.parameters.cytomine_id_terms),
            users=parse_list_or_none(cj.parameters.cytomine_id_users),
            reviewed=cj.parameters.cytomine_reviewed,
            showTerm=True, showMeta=True
        )

        cj.job.update(progress=5, statusComment="Download crops...")
        _ = annotations.dump_crops(
            dest_pattern=os.path.join(in_path, "{term}", "{image}_{id}.png"),
            n_workers=cj.parameters.n_jobs,
            override=False
        )

        cj.job.update(progress=15, statusComment="Prepare data loader....")
        width = 224
        batch_size = 16
        dataset = ImageFolderWithPaths(in_path, transform=normCenterCropTransform(width))
        sampler = BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cj.parameters.n_jobs)

        cj.job.update(progress=17, statusComment="Load network...")
        device = torch.device("cpu")
        n_features, net_fn = NETWORKS[cj.parameters.network]
        network = net_fn(pretrained=True)
        network.to(device)
        network.eval()

        n_samples = len(dataset)
        n_classes = len(dataset.classes)
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

        cj.logger.info("Data:")
        cj.logger.info("> nb images : {}".format(x.shape[0]))
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
        model_path = os.path.join(working_path, "model.pkl")
        joblib.dump(grid_search.best_estimator_, model_path)

        Property(cj.job, key="classifier", value=cj.parameters.classifier).save()
        Property(cj.job, key="network", value=cj.parameters.network).save()

        AttachedFile(
            cj.job,
            domainIdent=cj.job.id,
            filename=model_path,
            domainClassName="be.cytomine.processing.Job"
        ).upload()

        cj.job.update(status=Job.SUCCESS, statusComment="Finished.", progress=100)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
