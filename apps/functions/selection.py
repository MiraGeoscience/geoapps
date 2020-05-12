from ipywidgets import Dropdown, VBox, HBox, interactive_output, SelectMultiple
from .geoh5py.workspace import Workspace
from .plotting import plot_plan_data_selection


def object_data_selection_widget(h5file, plot=False, interactive=False, select_multiple=False):
    """

    """
    workspace = Workspace(h5file)

    def listObjects(obj_name, data_name):
        obj = workspace.get_entity(obj_name)[0]
        data = obj.get_data(data_name)[0]

        if plot:
            plot_plan_data_selection(obj, data)

        return obj, data

    names = list(workspace.list_objects_name.values())

    def updateList(_):
        workspace = Workspace(h5file)
        obj = workspace.get_entity(objects.value)[0]
        data.options = obj.get_data_list()

    objects = Dropdown(
        options=names,
        value=names[0],
        description='Object:',
    )

    obj = workspace.get_entity(objects.value)[0]

    if select_multiple:
        data = SelectMultiple(
            options=obj.get_data_list(),
            value=[obj.get_data_list()[0]],
            description='Data: ',
        )
    else:
        data = Dropdown(
            options=obj.get_data_list(),
            value=obj.get_data_list()[0],
            description='Data: ',
        )

    objects.observe(updateList, names='value')

    out = HBox([
        VBox([objects, data]),
        interactive_output(
            listObjects, {
                "obj_name": objects,
                "data_name": data
            }
        )
    ])

    if interactive:
        return out
    else:
        return objects, data