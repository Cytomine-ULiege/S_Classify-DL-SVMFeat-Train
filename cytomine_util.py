from cytomine.models import AnnotationCollection


def parse_list(s, _type=int, sep=","):
    return [_type(i.strip()) for i in s.split(sep)]


def parse_list_or_none(s, _type=int, sep=","):
    return None if s is None else parse_list(s, _type=_type, sep=sep)


def get_annotations(project_id, images=None, terms=None, users=None, reviewed=False, **kwargs):
    annotations = AnnotationCollection(
        filters={"project": project_id},
        images=images, term=terms, users=users,
        showTerm=True, **kwargs
    ).fetch()

    if reviewed:
        annotations += AnnotationCollection(
            filters={"project": project_id},
            images=images, terms=terms, users=users, reviewed=True,
            showTerm=True, **kwargs
        ).fetch()

    return annotations


