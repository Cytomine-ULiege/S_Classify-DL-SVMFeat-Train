import os
from cytomine import CytomineJob
import pickle
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.utils import check_random_state

from keras_util import MODEL_RESNET50, MODEL_VGG19, MODEL_VGG16, MODEL_INCEPTION_V3, MODEL_INCEPTION_RESNET_V2, \
    MODEL_MOBILE, MODEL_DENSE_NET_201, MODEL_NASNET_LARGE, MODEL_NASNET_MOBILE, PretrainedModelFeatures, ImageLoader, \
    batch_transform
from cytomine_util import parse_list_or_none, get_annotations

from sklearn.svm import LinearSVC


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        random_state = check_random_state(cj.parameters.random_seed)

        # prepare paths
        working_path = cj.parameters.working_directory
        data_path = os.path.join(working_path, "train_data")
        save_path = cj.parameters.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # load and dump annotations
        cj.job.update(statusComment="Download annotations.")
        annotations = get_annotations(
            project_id=cj.parameters.cytomine_project_id,
            images=parse_list_or_none(cj.parameters.cytomine_images_ids),
            terms=parse_list_or_none(cj.parameters.cytomine_terms_ids),
            users=parse_list_or_none(cj.parameters.cytomine_users_ids)
        )

        cj.job.update(statusComment="Fetch crops.", progress=10)
        n_samples = len(annotations)
        x = np.zeros([n_samples], dtype=np.object)
        y = np.zeros([n_samples], dtype=int)
        labels = np.zeros([n_samples], dtype=int)
        for i, annotation in cj.monitor(enumerate(annotations), start=10, end=40, prefix="Fetch crops", period=0.1):
            file_format = os.path.join(data_path, str(annotation.term[0]), "{id}.png")
            if not annotation.dump(dest_pattern=file_format):
                raise ValueError("Download error for annotation '{}'.".format(annotation.id))
            x[i] = file_format.format(id=annotation.id)
            y[i] = annotation.term[0]
            labels[i] = annotation.image

        available_nets = {
            MODEL_RESNET50, MODEL_VGG19, MODEL_VGG16, MODEL_INCEPTION_V3,
            MODEL_INCEPTION_RESNET_V2, MODEL_MOBILE, MODEL_DENSE_NET_201,
            MODEL_NASNET_LARGE, MODEL_NASNET_MOBILE
        }
        if cj.parameters.network not in available_nets:
            raise ValueError("Invalid value (='{}'} for parameter 'network'.".format(cj.parameters.network))
        if cj.parameters.reduction not in {"average_pooling"}:
            raise ValueError("Invalid value (='{}') for parameter 'reduction'.".format(cj.parameters.reduction))
        if cj.parameters.classifier not in {"svm"}:
            raise ValueError("Invalid value (='{}') for parameter 'classifier'.".format(cj.parameters.classifier))

        # prepare network
        cj.job.update(statusComment="Load neural network '{}'".format(cj.parameters.network), progress=40)
        features = PretrainedModelFeatures(
            model=cj.parameters.network, layer="last",
            reduction=cj.parameters.reduction, weights="imagenet"
        )
        height, width, _ = features._get_input_shape(cj.parameters.network)
        loader = ImageLoader(load_size_range=(height, height), crop_size=height, random_crop=False)

        cj.job.update(statusComment="Transform features.", progress=50)
        x_feat = batch_transform(loader, features, x, logger=cj.logger(start=50, end=70, period=0.1), batch_size=128)

        cj.job.update(statusComment="Prepare cross-validation of '{}'.".format(cj.parameters.classifier), progress=70)
        # prepare cross-validation strategy
        unique_labels = np.unique(labels)
        n_splits = min(cj.parameters.cv_folds, unique_labels.shape[0])
        if n_splits == 1:
            cv = KFold(n_splits=min(cj.parameters.cv_folds, n_samples), shuffle=True, random_state=random_state)
        else:
            cv = GroupKFold(n_splits=n_splits)

        if cj.parameters.classifier == "svm":
            model = LinearSVC()
            grid = {"C": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]}
        else:
            raise ValueError("No such classifier '{}'.".format(cj.parameters.classifier))

        grid_search = GridSearchCV(
            model, grid, cv=cv, scoring=make_scorer(accuracy_score),
            verbose=10, n_jobs=cj.parameters.n_jobs if cj.parameters.classifier == "svm" else 1,
            refit=True
        )

        cj.job.update(statusComment="Start grid search.", progress=80)
        grid_search.fit(x_feat, y, groups=labels)

        model_path = os.path.join(save_path, "model.pkl")
        cj.job.update(statusComment="Save model in '{}'.".format(model_path), progress=90)

        with open(model_path, "wb+") as file:
            pickle.dump({
                "model": grid_search.best_estimator_,
                "classifier": cj.parameters.classifier,
                "reduction": cj.parameters.reduction,
                "network": cj.parameters.network
            }, file)

        cj.job.update(statusComment="Finished.", progress=100)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
