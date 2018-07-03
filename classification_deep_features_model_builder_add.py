from cytomine import Cytomine
from cytomine.models import Software, SoftwareParameter


def main(argv):
    with Cytomine.connect_from_cli(argv):
        software = Software(name="Classification_Deep_Features_Model_Builder",
                            service_name="pyxitSuggestedTermJobService",
                            result_name="ValidateAnnotation").save()

        SoftwareParameter("cytomine_project_id", type="Number", id_software=software.id,
                          index=100, set_by_server=True, required=True).save()
        SoftwareParameter("cytomine_software_id", type="Number", id_software=software.id, default_value=software.id,
                          index=200, set_by_server=True, required=True).save()

        # filtering annotations
        SoftwareParameter("cytomine_images_ids", type="List", id_software=software.id, default_value=None, index=500).save()  # ids of images to use for training
        SoftwareParameter("cytomine_terms_ids", type="List", id_software=software.id, default_value=None, index=600).save()
        SoftwareParameter("cytomine_users_ids", type="List", id_software=software.id, default_value=None, index=650).save()
        SoftwareParameter("cytomine_reviewed", type="Boolean", id_software=software.id, default_value=False, index=700).save()  # true for also including reviewed annotations

        # running parameters
        SoftwareParameter("working_directory", type="String", id_software=software.id, default_value="", index=800, required=True).save()
        SoftwareParameter("save_path", type="String", id_software=software.id, default_value="/tmp", index=900, required=True).save()
        SoftwareParameter("n_jobs", type="Number", id_software=software.id, default_value=1, index=1000, required=True).save()

        # deep features and training parameters
        SoftwareParameter("network", type="String", id_software=software.id, default_value="dense_net_201", index=1100, required=True).save()
        SoftwareParameter("reduction", type="String", id_software=software.id, default_value="average_pooling", index=1200, required=True).save()
        SoftwareParameter("classifier", type="String", id_software=software.id, default_value="svm", index=1300, required=True).save()
        SoftwareParameter("cv_folds", type="Number", id_software=software.id, default_value=10, index=1400, required=True).save()
        SoftwareParameter("random_seed", type="Number", id_software=software.id, default_value=42, index=1500, required=True).save()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
