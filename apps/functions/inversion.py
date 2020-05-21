import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ipywidgets as widgets
from ipywidgets.widgets import Label, Dropdown, Layout, VBox, HBox, Text


from .geoh5py.workspace import Workspace
from .geoh5py.objects import Curve, BlockModel, Octree, Surface, Grid2D, Points
import json
from .plotting import plot_profile_data_selection, plot_plan_data_selection
from .utils import find_value, rotate_xy


def pf_inversion_widget(
        h5file,
        plot_scatter_xy=None,
        resolution=50,
        inducing_field="50000, 90, 0",
        objects=None
):
    workspace = Workspace(h5file)

    units = {"Gravity": "g/cc", "Magnetics": "SI"}
    names = list(workspace.list_objects_name.values())

    dsep = os.path.sep
    inv_dir = dsep.join(
        os.path.dirname(os.path.abspath(h5file)).split(dsep)
    )
    if len(inv_dir) > 0:
        inv_dir += dsep
    else:
        inv_dir = os.getcwd() + dsep

    def update_data_list(_):

        if workspace.get_entity(objects.value):
            obj = workspace.get_entity(objects.value)[0]

            data.options = [name for name in obj.get_data_list() if "visual" not in name.lower()]
            if data.options:
                data.value = [data.options[0]]

            sensor_value.options = [name for name in obj.get_data_list() if "visual" not in name.lower()] + ["Vertices"]
            sensor_value.value = "Vertices"
            write.button_style = 'warning'
            run.button_style = 'danger'

    def update_data_options(_):
        if workspace.get_entity(objects.value):
            obj = workspace.get_entity(objects.value)[0]

            if survey_type.value == "Magnetics":
                data_type_list = ["tmi", 'bxx', "bxy", "bxz", 'byy', "byz", "bzz"]
            else:
                data_type_list = ["gz", 'gxx', "gxy", "gxz", 'gyy', "gyz", "gzz"]

            channel_specs = {}
            for channel in data.value:
                if obj.get_data(channel):

                    data_obj = obj.get_data(channel)[0]
                    values = np.abs(data_obj.values)
                    channel_specs[channel] = VBox([
                        Dropdown(
                            description='Data type',
                            options=data_type_list,
                            value=find_value(data_type_list, [channel])
                        ),
                        Text(
                            description='Uncertainty (%, floor): ',
                            value=f"0, {np.percentile(values[values > 2e-18], 5)}",
                            style={'description_width': 'initial'}
                        )
                    ])

            components.specs = channel_specs
            components.options = list(channel_specs.keys())

            if list(channel_specs.keys()):
                components.value = list(channel_specs.keys())[0]
                components_panel.children = [components, channel_specs[components.value]]

            write.button_style = 'warning'
            run.button_style = 'danger'

    def update_topo_list(_):
        if workspace.get_entity(topo_objects.value):
            obj = workspace.get_entity(topo_objects.value)[0]
            print(topo_objects.value)
            topo_value.options = [name for name in obj.get_data_list() if "visual" not in name.lower()] + ['Vertices']

            write.button_style = 'warning'
            run.button_style = 'danger'

    objects = Dropdown(
        options=names,
        value=objects,
        description='Object:',
    )

    objects.observe(update_data_list, names="value")

    data = widgets.SelectMultiple(
        description='Channels: ',
    )

    data.observe(update_data_options, names="value")

    components = widgets.Dropdown(
        description='Component Specs'
    )

    components_panel = VBox([components])

    def update_component_panel(_):
        if components.value is not None:
            components_panel.children = [components, components.specs[components.value]]
        else:
            components_panel.children = [components]

        write.button_style = 'warning'
        run.button_style = 'danger'

    components.observe(update_component_panel, names="value")

    ###################### Data selection ######################
    # Fetch vertices in the project
    lim_x = [1e+8, 0]
    lim_y = [1e+8, 0]
    for name in names:
        obj = workspace.get_entity(name)[0]
        if obj.vertices is not None:
            lim_x[0], lim_x[1] = np.min([lim_x[0], obj.vertices[:, 0].min()]), np.max(
                [lim_x[1], obj.vertices[:, 0].max()])
            lim_y[0], lim_y[1] = np.min([lim_y[0], obj.vertices[:, 1].min()]), np.max(
                [lim_y[1], obj.vertices[:, 1].max()])
        elif hasattr(obj, "centroids"):
            lim_x[0], lim_x[1] = np.min([lim_x[0], obj.centroids[:, 0].min()]), np.max(
                [lim_x[1], obj.centroids[:, 0].max()])
            lim_y[0], lim_y[1] = np.min([lim_y[0], obj.centroids[:, 1].min()]), np.max(
                [lim_y[1], obj.centroids[:, 1].max()])

    center_x = widgets.FloatSlider(
        min=lim_x[0], max=lim_x[1], value=np.mean(lim_x),
        steps=10, description="Easting", continuous_update=False
    )
    center_y = widgets.FloatSlider(
        min=lim_y[0], max=lim_y[1], value=np.mean(lim_y),
        steps=10, description="Northing", continuous_update=False,
        orientation='vertical',
    )
    azimuth = widgets.FloatSlider(
        min=-90, max=90, value=0, steps=5, description="Orientation", continuous_update=False
    )
    width_x = widgets.FloatSlider(
        max=lim_x[1] - lim_x[0],
        min=100,
        value=lim_x[1] - lim_x[0],
        steps=10, description="Width", continuous_update=False
    )
    width_y = widgets.FloatSlider(
        max=lim_y[1] - lim_y[0],
        min=100,
        value=lim_y[1] - lim_y[0],
        steps=10, description="Height", continuous_update=False,
        orientation='vertical'
    )
    resolution = widgets.FloatText(value=resolution, description="Resolution (m)",
                                   style={'description_width': 'initial'})

    data_count = Label("Data Count: 0", tooltip='Keep <1500 for speed')

    def plot_selection(
            entity_name, data_names, resolution,
            center_x, center_y,
            width_x, width_y, azimuth,
            zoom_extent
    ):
        if workspace.get_entity(entity_name):
            obj = workspace.get_entity(entity_name)[0]

            if len(data_names)>0 and obj.get_data(data_names[0]):
                fig = plt.figure(figsize=(10, 10))
                ax1 = plt.subplot()

                corners = np.r_[np.c_[-1., -1.], np.c_[-1., 1.], np.c_[1., 1.], np.c_[1., -1.], np.c_[-1., -1.]]
                corners[:, 0] *= width_x/2
                corners[:, 1] *= width_y/2
                corners = rotate_xy(corners, [0,0], -azimuth)
                ax1.plot(corners[:, 0] + center_x, corners[:, 1] + center_y, 'k')
                data_obj = obj.get_data(data_names[0])[0]
                _, ind_filter = plot_plan_data_selection(
                    obj, data_obj,
                    **{
                        "ax": ax1,
                        "downsampling": resolution,
                        "window": {
                            "center": [center_x, center_y],
                            "size": [width_x, width_y],
                            "azimuth": azimuth
                        },
                        "zoom_extent": zoom_extent
                    }
                )
                data_count.value = f"Data Count: {ind_filter.sum()}"
                # if plot_scatter_xy is not None:
                #     if isinstance(plot_scatter_x, np.ndarray):
                #         ax1.scatter(plot_scatter_xy[:, 0], plot_scatter_xy[:, 1], 5, 'k')
            write.button_style = 'warning'
            run.button_style = 'danger'

    zoom_extent = widgets.ToggleButton(
        value=False,
        description='Zoom on selection',
        tooltip='Keep plot extent on selection',
        icon='check'
    )

    plot_window = widgets.interactive_output(
        plot_selection, {
            "entity_name": objects,
            "data_names": data,
            "resolution": resolution,
            "center_x": center_x,
            "center_y": center_y,
            "width_x": width_x,
            "width_y": width_y,
            "azimuth": azimuth,
            "zoom_extent": zoom_extent
        }
    )

    selection_panel = VBox([
        Label("Window & Downsample"),
        VBox([resolution, data_count,
            HBox([
                center_y, width_y,
                plot_window,
            ], layout=Layout(align_items='center')),
            VBox([width_x, center_x, azimuth, zoom_extent], layout=Layout(align_items='center'))
        ], layout=Layout(align_items='center'))
    ])

    def update_survey_type(_):
        if survey_type.value == "Magnetics":
            survey_type_panel.children = [Label("Data"), survey_type, objects, data, components_panel, inducing_field]
        else:
            survey_type_panel.children = [Label("Data"), survey_type, objects, data, components_panel]

        update_data_options("")

        if ref_mod.children[1].children[1].children:
            ref_mod.children[1].children[1].children[0].decription = units[survey_type.value]

        write.button_style = 'warning'
        run.button_style = 'danger'

    survey_type = Dropdown(
        options=["Magnetics", "Gravity"],
        description="Survey Type:",
    )
    inducing_field = widgets.Text(
        value=inducing_field,
        description='Inducing Field [Amp, Inc, Dec]',
        style={'description_width': 'initial'}
    )
    survey_type.observe(update_survey_type)

    # data_type = Dropdown(
    #     description="Data Type:",
    # )

    survey_type_panel = VBox([survey_type, objects, data, components_panel])

    ###################### Spatial parameters ######################
    ########## TOPO #########
    topo_objects = Dropdown(
        options=names,
        value=find_value(names, ['topo', 'dem', 'dtm']),
        description='Object:',
    )
    topo_objects.observe(update_topo_list, names="value")

    topo_value = Dropdown(
        description='Channel: ',
    )

    topo_panel = VBox([topo_objects, topo_value])

    topo_offset = widgets.FloatText(
        value=-30,
        description="Vertical offset (m)",
        style={'description_width': 'initial'}
    )
    topo_options = {
        "Object": topo_panel,
        "Drape Height": topo_offset
    }

    topo_options_button = widgets.RadioButtons(
        options=['Object', 'Drape Height'],
        description='Define by:',
    )

    def update_topo_options(_):
        topo_options_panel.children = [topo_options_button, topo_options[topo_options_button.value]]
        write.button_style = 'warning'
        run.button_style = 'danger'

    topo_options_button.observe(update_topo_options)

    topo_options_panel = VBox([topo_options_button, topo_options[topo_options_button.value]])

    ########## RECEIVER #########
    sensor_value = Dropdown(
        options=["Vertices"],
        value="Vertices",
        description='Channel: ',
    )

    sensor_offset = widgets.FloatText(
        value=30,
        description="Vertical offset (m)",
        style={'description_width': 'initial'}
    )

    sensor_options = {
        "Channel": sensor_value,
        "Drape Height (Topo required)": sensor_offset
    }

    sensor_options_button = widgets.RadioButtons(
        options=['Channel', 'Drape Height (Topo required)'],
        description='Define by:',
    )

    def update_sensor_options(_):
        if topo_value.value is None:
            sensor_options_button.value = "Channel"

        sensor_options_panel.children = [sensor_options_button, sensor_options[sensor_options_button.value]]
        write.button_style = 'warning'
        run.button_style = 'danger'

    sensor_options_button.observe(update_sensor_options)

    sensor_options_panel = VBox([sensor_options_button, sensor_options[sensor_options_button.value]])

    ###############################
    spatial_options = {
        "Topography": topo_options_panel,
        "Sensor Height": sensor_options_panel
    }

    spatial_choices = widgets.Dropdown(
        options=list(spatial_options.keys()),
        value=list(spatial_options.keys())[0],
        disabled=False
    )

    spatial_panel = VBox(
        [Label("Spatial Information"), VBox([spatial_choices, spatial_options[spatial_choices.value]])])

    def spatial_option_change(_):
        spatial_panel.children[1].children = [spatial_choices, spatial_options[spatial_choices.value]]
        write.button_style = 'warning'
        run.button_style = 'danger'

    spatial_choices.observe(spatial_option_change)

    update_data_list("")

    ###################### Inversion options ######################
    def write_unclick(_):
        if write.value:
            input_dict = {}
            input_dict['out_group'] = out_group.value
            input_dict['workspace'] = h5file
            input_dict['save_to_geoh5'] = h5file
            if survey_type.value == 'Gravity':
                input_dict["inversion_type"] = 'grav'
            elif survey_type.value == "Magnetics":
                input_dict["inversion_type"] = 'mvis'
                input_dict["inducing_field_aid"] = np.asarray(inducing_field.value.split(",")).astype(float).tolist()

            uncertainties = []
            channels = []
            comps = []
            for channel, specs in components.specs.items():
                channels.append(channel)
                uncertainties.append(np.asarray(specs.children[1].value.split(",")).astype(float).tolist())
                comps.append(specs.children[0].value)

            input_dict["data_type"] = {
                "GA_object": {
                    "name": objects.value,
                    "data": channels,
                    "components": comps,
                    "uncertainties": uncertainties
                }
            }

            if sensor_options_button.value == 'Channel':
                input_dict["data_type"]["GA_object"]['z_channel'] = sensor_value.value
            else:
                input_dict['drape_data'] = sensor_offset.value

            if topo_options_button.value == "Object":
                input_dict["topography"] = {
                    "GA_object": {
                        "name": topo_objects.value,
                        "data": topo_value.value,
                    }
                }
            else:
                input_dict["topography"] = {"drapped": topo_offset}

            input_dict['resolution'] = resolution.value
            input_dict["window"] = {
                        "center": [center_x.value, center_y.value],
                        "size": [width_x.value, width_y.value],
                        "azimuth": azimuth.value
            }

            input_dict["alphas"] = np.asarray(alpha_values.value.split(",")).astype(float).tolist()

            if ref_type.value != "None":
                input_dict["model_reference"] = ref_mod.children[1].children[1].children[0].value
            else:
                input_dict["alphas"][0] = 0

            input_dict["model_norms"] = np.asarray(norms.value.split(",")).astype(float).tolist()
            # input_dict["core_cell_size"] = [resolution.value/2, resolution.value/2, resolution.value/2]
            # input_dict["octree_levels_topo"] = [0, 0, 3]
            # input_dict["octree_levels_obs"] = [8, 8, 8]
            input_dict["core_cell_size"] = np.asarray(core_cell_size.value.split(",")).astype(float).tolist()
            input_dict["octree_levels_topo"] = np.asarray(octree_levels_topo.value.split(",")).astype(float).tolist()
            input_dict["octree_levels_obs"] = np.asarray(octree_levels_obs.value.split(",")).astype(float).tolist()
            input_dict['depth_core'] = {'value': depth_core.value}

            # input_dict["octree_levels_padding"] = [5, 5, 5]
            input_dict["padding_distance"] = [
                [width_x.value/2, width_x.value/2],
                [width_y.value / 2, width_y.value / 2],
                [np.min([width_x.value / 2, width_y.value / 2]), 0]
            ]
            # input_dict['depth_core'] = {'auto': 0.5}

            if forward_only.value:
                input_dict['forward_only'] = []

            with open(inv_dir + f"{out_group.value}.json", 'w') as f:
                json.dump(input_dict, f)

            write.value = False
            write.button_style = ''
            run.button_style = 'success'

    def run_unclick(_):
        if run.value:
            prompt = os.system(
                "start cmd.exe @cmd /k " + f"\"python functions/pf_inversion.py {inv_dir}{out_group.value}.json\"")
            run.value = False
            run.button_style = ''

    forward_only = widgets.Checkbox(
        value=False,
        description="Forward only",
        tooltip='Forward response of reference model',
    )

    def update_options(_):
        write.button_style = 'warning'
        run.button_style = 'danger'

    forward_only.observe(update_options)

    run = widgets.ToggleButton(
        value=False,
        description='Run SimPEG',
        button_style='danger',
        icon='check'
    )

    run.observe(run_unclick)

    out_group = widgets.Text(
        value='Inversion_',
        description='Save to:',
        disabled=False
    )

    write = widgets.ToggleButton(
        value=False,
        description='Write input',
        button_style='warning',
        tooltip='Write json input file',
        icon='check'
    )

    write.observe(write_unclick)

    chi_factor = widgets.FloatText(
        value=1,
        description='Target misfit',
        disabled=False
    )
    chi_factor.observe(update_options)
    ref_type = widgets.RadioButtons(
        options=["None", 'Value', 'Model'],
        value='Value',
        disabled=False
    )
    ref_type.observe(update_options)
    alpha_values = widgets.Text(
        value='1, 1, 1, 1',
        description='Regularization (m, x, y, z)',
        disabled=False,
        style={'description_width': 'initial'}
    )
    alpha_values.observe(update_options)
    norms = widgets.Text(
        value='2, 2, 2, 2',
        description='Norms (m, x, y, z)',
        disabled=False,
        style={'description_width': 'initial'}
    )
    norms.observe(update_options)
    def update_ref(_):

        if ref_mod.children[1].children[0].value == 'Model':

            model_list = []

            for obj in workspace.all_objects():
                if isinstance(obj, BlockModel) or isinstance(obj, Octree):

                    for data in obj.children:

                        if getattr(data, "values", None) is not None:
                            model_list += [data.name]

            ref_mod.children[1].children[1].children = [widgets.Dropdown(
                description='3D Model',
                options=model_list,
            )]
            alpha_values.value= '1, 1, 1, 1'

        elif ref_mod.children[1].children[0].value == 'Value':

            ref_mod.children[1].children[1].children = [widgets.FloatText(
                description=units[survey_type.value],
                value=0.,
            )]
            alpha_values.value = '1, 1, 1, 1'
        else:
            ref_mod.children[1].children[1].children = []
            alpha_values.value = '0, 1, 1, 1'

        write.button_style = 'warning'
        run.button_style = 'danger'


    ref_type.observe(update_ref)
    ref_mod = widgets.VBox([
        Label('Reference model'),
        widgets.VBox([ref_type, widgets.VBox([
            widgets.FloatText(
                description=units[survey_type.value],
                value=0.,
            )
        ])])])

    core_cell_size = widgets.Text(
        value='25, 25, 25',
        description='Smallest cells',
        disabled=False,
        style={'description_width': 'initial'}
    )

    octree_levels_topo = widgets.Text(
        value='0, 0, 3',
        description='Layers below topo',
        disabled=False,
        style={'description_width': 'initial'}
    )

    octree_levels_obs = widgets.Text(
        value='8, 8, 8',
        description='Layers below data',
        disabled=False,
        style={'description_width': 'initial'}
    )

    depth_core = widgets.FloatText(
        value=500,
        description='Core depth (m)',
        disabled=False,
        style={'description_width': 'initial'}
    )

    mesh_panel = widgets.VBox([
        Label('Octree Mesh'),
        widgets.VBox([
            core_cell_size,
            octree_levels_topo,
            octree_levels_obs,
            depth_core
        ])
    ])

    def update_octree_param(_):

        core_cell_size.value = f"{resolution.value}, {resolution.value}, {resolution.value}"
        depth_core.value = np.min([width_x.value, width_y.value])/2.

    width_x.observe(update_octree_param)
    width_y.observe(update_octree_param)
    resolution.observe(update_octree_param)

    inversion_options = {
        "output name": out_group,
        "target misfit": chi_factor,
        "reference model": ref_mod,
        "regularization": alpha_values,
        "norms": norms,
        "octree mesh": mesh_panel
    }

    option_choices = widgets.Dropdown(
        options=list(inversion_options.keys()),
        value=list(inversion_options.keys())[0],
        disabled=False
    )

    def inv_option_change(_):
        inversion_panel.children[1].children = [option_choices, inversion_options[option_choices.value]]

    option_choices.observe(inv_option_change)

    inversion_panel = widgets.VBox([
        widgets.HBox([widgets.Label("Inversion Options")]),
        widgets.HBox([option_choices, inversion_options[option_choices.value]], )
        # layout=widgets.Layout(height='500px')
    ], layout=Layout(width="100%"))

    survey_type.value = "Gravity"
    update_topo_list("")

    return VBox([
        HBox([survey_type_panel, spatial_panel]),
        selection_panel, inversion_panel, forward_only, write, run
    ])


def em1d_inversion_widget(h5file, plot_profile=True, start_channel=None, object_name=None):
    """
    Setup and invert time or frequency domain data
    """
    workspace = Workspace(h5file)

    curves = [entity.parent.name + "." + entity.name for entity in workspace.all_objects() if isinstance(entity, Curve)]
    names = [name for name in sorted(curves)]

    all_obj = [entity.parent.name + "." + entity.name for entity in workspace.all_objects() if isinstance(
        entity, (Curve, Grid2D, Surface, Points)
    )]

    all_names = [name for name in sorted(all_obj)]

    # Load all known em systems
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "AEM_systems.json"), 'r') as aem_systems:
        em_system_specs = json.load(aem_systems)

    dsep = os.path.sep
    inv_dir = dsep.join(
        os.path.dirname(os.path.abspath(h5file)).split(dsep)
    )
    if len(inv_dir) > 0:
        inv_dir += dsep
    else:
        inv_dir = os.getcwd() + dsep

    def get_parental_child(parental_name):
        if parental_name is not None:
            parent, child = parental_name.split(".")

            parent_entity = workspace.get_entity(parent)[0]

            children = [entity for entity in parent_entity.children if entity.name==child]
            return children
        return None

    def find_value(labels, strings):
        value = None
        for name in labels:
            for string in strings:
                if string.lower() in name.lower():
                    value = name
        return value

    def get_comp_list(entity):
        component_list = []

        for pg in entity.property_groups:
            component_list.append(pg.name)

        return component_list

    objects = Dropdown(
        options=names,
        description='Object:',
    )
    def object_observer(_):
        if get_parental_child(objects.value):
            entity = get_parental_child(objects.value)[0]
            data_list = entity.get_data_list()

            # Update topo field
            # topo_value.options = data_list
            # topo_value.value = find_value(data_list, ['dem', "topo", 'dtm'])

            sensor_value.options = [name for name in data_list if "visual" not in name.lower()] + ["Vertices"]

            line_field.options = data_list
            line_field.value = find_value(data_list, ['line'])

            if get_comp_list(entity):
                components.options = get_comp_list(entity)
                components.value = [get_comp_list(entity)[0]]

            for aem_system, specs in em_system_specs.items():
                if any([specs["flag"] in channel for channel in data_list]):
                    system.value = aem_system

            system_observer("")
            line_field_observer("")
            write.button_style = 'warning'
            invert.button_style = 'danger'

    objects.observe(object_observer, names='value')

    systems = list(em_system_specs.keys())
    system = Dropdown(
        options=systems,
        description='System: ',
    )

    scale = Dropdown(
        options=['linear', 'symlog'],
        value='symlog',
        description='Scaling',
    )

    def system_observer(_, start_channel=start_channel):
        entity = get_parental_child(objects.value)[0]
        rx_offsets = em_system_specs[system.value]["rx_offsets"]
        bird_offset.value = ', '.join([
                                str(offset) for offset in em_system_specs[system.value]["bird_offset"]
                            ])
        uncertainties = em_system_specs[system.value]["uncertainty"]

        if start_channel is None:
            start_channel = em_system_specs[system.value]["channel_start_index"]

        if em_system_specs[system.value]["type"] == 'time':
            label = "Time (s)"
            scale.value = "symlog"
        else:
            label = "Frequency (Hz)"
            scale.value = "linear"

        data_channel_options = {}
        for ind, (key, time) in enumerate(
            em_system_specs[system.value]["channels"].items()
        ):
            if ind+1 < start_channel:
                continue

            if len(rx_offsets) > 1:
                offsets = rx_offsets[ind]
            else:
                offsets = rx_offsets[0]

            data_list = []
            if components.value is not None:
                for component in components.value:
                    p_g = entity.get_property_group(component)
                    if p_g is not None:
                        data_list += (
                            [workspace.get_entity(data)[0].name for data in p_g.properties]
                        )
            if len(data_list) == 0:
                data_list = entity.get_data_list()

            data_channel_options[key] = VBox([
                        widgets.Checkbox(
                            value=ind+1 >= start_channel,
                            indent=True,
                            description="Active"
                        ),
                        widgets.Text(
                            value=f"{time:.5e}",
                            description=label,
                            style={'description_width': 'initial'}
                        ),
                        Dropdown(
                            options=data_list,
                            value=find_value(data_list, [key]),
                            description="Channel",
                            style={'description_width': 'initial'}
                        ),
                        widgets.Text(
                            value=', '.join([
                                str(uncert) for uncert in uncertainties[ind][:2]
                            ]),
                            description="Error (%, floor)",
                            style={'description_width': 'initial'}
                        ),
                        widgets.Text(
                            value=', '.join([
                                str(offset) for offset in offsets
                            ]),
                            description='Offset',
                            style={'description_width': 'initial'}
                        )
                ])

        data_channel_choices.options = list(data_channel_options.keys())
        data_channel_choices.value = list(data_channel_options.keys())[0]
        system.data_channel_options = data_channel_options
        data_channel_panel.children = [
            data_channel_choices,
            data_channel_options[data_channel_choices.value]
        ]
        write.button_style = 'warning'
        invert.button_style = 'danger'

    system.observe(system_observer, names='value')

    components = widgets.SelectMultiple(
        description='Data Groups:',
    )

    def data_channel_choices_observer(_):
        if (
            hasattr(system, "data_channel_options") and
            data_channel_choices.value in (system.data_channel_options.keys())
        ):
            data_channel_panel.children = [data_channel_choices, system.data_channel_options[data_channel_choices.value]]
        write.button_style = 'warning'
        invert.button_style = 'danger'

    data_channel_choices = widgets.Dropdown(
            description="Data field:",
            style={'description_width': 'initial'}
        )

    data_channel_choices.observe(data_channel_choices_observer, names='value')
    data_channel_panel = widgets.VBox([data_channel_choices])  #layout=widgets.Layout(height='500px')

    def auto_pick_channels(_):
        entity = get_parental_child(objects.value)[0]

        if components.value is not None:
            data_list = []
            for component in components.value:
                p_g = entity.get_property_group(component)
                if p_g is not None:
                    data_list += (
                        [workspace.get_entity(data)[0].name for data in p_g.properties]
                    )
        else:
            data_list = entity.get_data_list()

        if hasattr(system, "data_channel_options"):
            for key, data_widget in system.data_channel_options.items():

                value = find_value(data_list, [key])

                data_widget.children[2].options = data_list
                data_widget.children[2].value = value

    components.observe(auto_pick_channels, names='value')

    ###################### Spatial parameters ######################
    ########## TOPO #########
    def update_topo_list(_):

        if get_parental_child(topo_objects.value):
            obj = get_parental_child(topo_objects.value)[0]
            topo_value.options = [name for name in obj.get_data_list() if "visual" not in name.lower()] + ['Vertices']
            topo_value.value = find_value(topo_value.options, ['dem', "topo", 'dtm'])

            write.button_style = 'warning'
            invert.button_style = 'danger'

    topo_objects = Dropdown(
        options=all_names,
        value=find_value(names, ['topo', 'dem', 'dtm']),
        description='Object:',
    )
    topo_objects.observe(update_topo_list, names="value")

    topo_value = Dropdown(
        description='Channel: ',
    )

    topo_panel = VBox([topo_objects, topo_value])

    topo_offset = widgets.FloatText(
        value=-30,
        description="Vertical offset (m)",
        style={'description_width': 'initial'}
    )
    topo_options = {
        "Object": topo_panel,
        "Drape Height": topo_offset
    }

    topo_options_button = widgets.RadioButtons(
        options=['Object', 'Drape Height'],
        description='Define by:',
    )

    def update_topo_options(_):
        topo_options_panel.children = [topo_options_button, topo_options[topo_options_button.value]]
        write.button_style = 'warning'
        invert.button_style = 'danger'

    topo_options_button.observe(update_topo_options)

    topo_options_panel = VBox([topo_options_button, topo_options[topo_options_button.value]])

    ########## RECEIVER #########
    sensor_value = Dropdown(
        description='Channel: ',
    )

    bird_offset = Text(
        description="[dx, dy, dz]"
    )

    sensor_options = {
        "Locations + offset (m)": bird_offset,
        "Topo + offset": bird_offset,
        "Topo + radar": sensor_value
    }

    sensor_options_button = widgets.RadioButtons(
        options=["Locations + offset (m)", "Topo + offset", "Topo + radar"],
        description='Define by:',
    )

    def update_sensor_options(_):
        if topo_value.value is None:
            sensor_options_button.value = "Locations + offset (m)"

        sensor_options_panel.children = [sensor_options_button, sensor_options[sensor_options_button.value]]
        write.button_style = 'warning'
        invert.button_style = 'danger'

    sensor_options_button.observe(update_sensor_options)

    sensor_options_panel = VBox([sensor_options_button, sensor_options[sensor_options_button.value]])

    ###############################
    spatial_options = {
        "Topography": topo_options_panel,
        "Sensor Height": sensor_options_panel
    }

    line_field = Dropdown(
        description='Lines field',
    )

    def line_field_observer(_):
        if (
            objects.value is not None and
            line_field.value is not None and
            'line' in line_field.value.lower()
        ):
            entity = get_parental_child(objects.value)[0]
            if entity.get_data(line_field.value):
                lines.options = np.unique(
                    entity.get_data(line_field.value)[0].values
                )

                if lines.options[1]:
                    lines.value = [lines.options[1]]
                if lines.options[0]:
                    lines.value = [lines.options[0]]
            write.button_style = 'warning'
            invert.button_style = 'danger'

    line_field.observe(line_field_observer, names='value')
    lines = widgets.SelectMultiple(
        description=f"Select data:",
    )
    downsampling = widgets.FloatText(
        value=0,
        description='Downsample (m)',
        style={'description_width': 'initial'}
    )

    object_fields_options = {
        "Data Channels": data_channel_panel,
        "Topography": topo_options_panel,
        "Sensor Height": sensor_options_panel,
        "Line ID": line_field,
    }

    object_fields_dropdown = widgets.Dropdown(
            options=list(object_fields_options.keys()),
            value=list(object_fields_options.keys())[0],
    )

    object_fields_panel = widgets.VBox([
        widgets.VBox([object_fields_dropdown], layout=widgets.Layout(height='75px')),
        widgets.VBox([object_fields_options[object_fields_dropdown.value]])  #layout=widgets.Layout(height='500px')
    ], layout=widgets.Layout(height='225px'))

    def object_fields_panel_change(_):
        object_fields_panel.children[1].children = [object_fields_options[object_fields_dropdown.value]]
        write.button_style = 'warning'
        invert.button_style = 'danger'

    object_fields_dropdown.observe(object_fields_panel_change, names='value')

    def get_fields_list(field_dict):
        plot_field = []
        for field_widget in field_dict.values():
            if field_widget.children[0].value:
                plot_field.append(field_widget.children[2].value)

        return plot_field

    def fetch_uncertainties():
        uncerts = []
        if hasattr(system, "data_channel_options"):
            for key, data_widget in system.data_channel_options.items():
                if data_widget.children[0].value:
                    uncerts.append(np.asarray(data_widget.children[3].value.split(",")).astype(float))

        return uncerts

    def show_selection(
        line_ids, downsampling, plot_uncert, scale
    ):
        workspace = Workspace(h5file)

        if get_parental_child(objects.value):
            entity = get_parental_child(objects.value)[0]

            if plot_uncert:
                uncerts = fetch_uncertainties()
            else:
                uncerts = None

            locations = entity.vertices
            parser = np.ones(locations.shape[0], dtype='bool')

            if hasattr(system, "data_channel_options"):

                fig = plt.figure(figsize=(12, 8))
                ax1 = plt.subplot(2, 1, 1)
                ax2 = plt.subplot(2, 1, 2)

                plot_field = get_fields_list(system.data_channel_options)

                if entity.get_data(plot_field[0]):
                    data = entity.get_data(plot_field[0])[0]

                    plot_plan_data_selection(
                            entity, data, **{
                                "highlight_selection": {line_field.value: line_ids},
                                "downsampling": downsampling,
                                "ax": ax1,
                                "color_norm": colors.SymLogNorm(
                                    linthresh=np.percentile(np.abs(data.values), 10), base=10
                                )
                            }
                    )

                    if plot_profile:
                        ax2, threshold = plot_profile_data_selection(
                            entity, plot_field,
                            selection={line_field.value: line_ids},
                            downsampling=downsampling,
                            uncertainties=uncerts,
                            ax=ax2
                        )

                        if scale == 'linear':
                            plt.yscale(scale)
                        else:
                            plt.yscale(scale, linthreshy=threshold)

            write.button_style = 'warning'
            invert.button_style = 'danger'

    if em_system_specs[system.value]['type'] == 'time':
        uncert_type ='Estimated (%|data| + background)'
    else:
        uncert_type = 'User input (\%|data| + floor)'

    uncert_mode = widgets.RadioButtons(
        options=['Estimated (%|data| + background)', 'User input (\%|data| + floor)'],
        value=uncert_type,
        disabled=False
    )

    # uncert_mode.observe(uncert_values_active)

    uncert_panel = widgets.VBox(
        [Label("Apply to:"), uncert_mode],
        layout=widgets.Layout(width='50%')
    )

    plot_uncert = widgets.Checkbox(
        value=False,
        description="Plot uncertainties"
        # indent=False
    )

    data_selection_panel = VBox([
        lines, downsampling, plot_uncert, scale
    ], layout=Layout(width="50%"))

    interactive_plot = widgets.interactive_output(
                show_selection, {
                    "line_ids": lines,
                    "downsampling": downsampling,
                    "plot_uncert": plot_uncert,
                    "scale": scale,
                }
    )
    data_panel = widgets.HBox([
            data_selection_panel,
            HBox([interactive_plot], layout=Layout(width="50%"))]
    )

    ############# Inversion panel ###########
    def write_unclick(_):
        if write.value:
            workspace = Workspace(h5file)
            entity = get_parental_child(objects.value)[0]
            input_dict = {}
            input_dict["system"] = system.value

            if sensor_options_button.value == "Locations + offset (m)":
                input_dict['rx_absolute'] = np.asarray(bird_offset.value.split(",")).astype(float).tolist()
            elif sensor_options_button.value == "Topo + offset":
                input_dict['rx_relative_drape'] = np.asarray(bird_offset.value.split(",")).astype(float).tolist()
            else:
                input_dict['rx_relative_radar'] = sensor_value.value

            if topo_options_button.value == "Object":

                if topo_objects.value is None:
                    input_dict["topography"] = None
                else:
                    input_dict["topography"] = {
                        "GA_object": {
                            "name": topo_objects.value,
                            "data": topo_value.value,
                        }
                    }
            else:
                input_dict["topography"] = {"drapped": topo_offset}

            input_dict['workspace'] = h5file
            input_dict['entity'] = entity.name
            input_dict['lines'] = {line_field.value: [str(line) for line in lines.value]}
            input_dict['downsampling'] = str(downsampling.value)
            input_dict['chi_factor'] = [chi_factor.value]
            input_dict['out_group'] = out_group.value
            input_dict["model_norms"] = np.asarray(norms.value.split(",")).astype(float).tolist()
            input_dict["mesh 1D"] = [hz_min.value, hz_expansion.value, n_cells.value]
            input_dict["ignore values"] = ignore_values.value
            input_dict["iterations"] = max_iteration.value
            input_dict["bounds"] = [lower_bound.value, upper_bound.value]

            if ref_mod.children[1].children[1].children:
                input_dict['reference'] = ref_mod.children[1].children[1].children[0].value
            else:
                input_dict['reference'] = []

            if start_mod.children[1].children[1].children:
                input_dict['starting'] = start_mod.children[1].children[1].children[0].value
            else:
                input_dict['starting'] = []

            if susc_type.value != 'None':
                input_dict['susceptibility'] = susc_mod.children[1].children[1].children[0].value

            input_dict["data"] = {}

            input_dict["uncert"] = {"mode": uncert_mode.value}
            input_dict["uncert"]['channels'] = {}

            if em_system_specs[system.value]['type'] == 'time' and hasattr(system, "data_channel_options"):
                data_widget = list(system.data_channel_options.values())[0]
                input_dict['rx_offsets'] = np.asarray(data_widget.children[4].value.split(",")).astype(float).tolist()
            else:
                input_dict['rx_offsets'] = {}

            if hasattr(system, "data_channel_options"):
                for key, data_widget in system.data_channel_options.items():
                    if data_widget.children[0].value:
                        input_dict["data"][key] = data_widget.children[2].value
                        input_dict["uncert"]['channels'][key] = np.asarray(data_widget.children[3].value.split(",")).astype(float).tolist()

                        if em_system_specs[system.value]['type'] == 'frequency':
                            input_dict['rx_offsets'][key] = np.asarray(data_widget.children[4].value.split(",")).astype(float).tolist()

            input_check = [key for key, val in input_dict.items() if val is None]
            if len(input_check) > 0:
                print(f"Required value for {input_check}")
                invert.button_style = 'danger'
            else:
                write.button_style = ''
                with open(inv_dir + f"{out_group.value}.json", 'w') as f:
                    json.dump(input_dict, f)
                invert.button_style = 'success'

            write.value = False

    def invert_unclick(_):
        if invert.value:

            if em_system_specs[system.value]['type'] == 'time':
                prompt = os.system("start cmd.exe @cmd /k " + f"\"python functions/tem1d_inversion.py {inv_dir}{out_group.value}.json\"")
            else:
                prompt = os.system("start cmd.exe @cmd /k " + f"\"python functions/fem1d_inversion.py {inv_dir}{out_group.value}.json\"")

            invert.value = False

    model_list = []
    for obj in workspace.all_objects():
        if isinstance(obj, (BlockModel, Octree, Surface)):
            for data in obj.children:
                if getattr(data, "values", None) is not None:
                    model_list += [data.name]

    def update_ref(_):

        if ref_mod.children[1].children[0].value == 'Best-fitting halfspace':

            ref_mod.children[1].children[1].children = []

        elif ref_mod.children[1].children[0].value == 'Model':
            ref_mod.children[1].children[1].children = [ref_mod_list]
        else:
            ref_mod.children[1].children[1].children = [ref_mod_value]
        write.button_style = 'warning'
        invert.button_style = 'danger'

    def update_start(_):

        if start_mod.children[1].children[0].value == 'Model':
            start_mod.children[1].children[1].children = [start_mod_list]
        else:
            start_mod.children[1].children[1].children = [start_mod_value]

        write.button_style = 'warning'
        invert.button_style = 'danger'

    def update_susc(_):

        if susc_mod.children[1].children[0].value == 'None':
            susc_mod.children[1].children[1].children = []

        elif susc_mod.children[1].children[0].value == 'Model':
            susc_mod.children[1].children[1].children = [susc_mod_list]

        else:
            susc_mod.children[1].children[1].children = [start_mod_value]
        write.button_style = 'warning'
        invert.button_style = 'danger'

    invert = widgets.ToggleButton(
        value=False,
        description='Invert',
        button_style='danger',
        tooltip='Run simpegEM1D',
        icon='check'
    )

    invert.observe(invert_unclick)

    out_group = widgets.Text(
        value='Inversion_',
        description='Save to:',
        disabled=False
    )

    write = widgets.ToggleButton(
        value=False,
        description='Write input',
        button_style='',
        tooltip='Write json input file',
        icon='check'
    )

    write.observe(write_unclick)

    chi_factor = widgets.FloatText(
        value=1,
        description='Target misfit',
        disabled=False
    )

    max_iteration = widgets.IntText(
        value=10,
    )

    ref_type = widgets.RadioButtons(
        options=['Best-fitting halfspace', 'Model', 'Value'],
        value='Best-fitting halfspace',
        disabled=False
    )

    ref_type.observe(update_ref)

    ref_mod_list = widgets.Dropdown(
        description='3D Model',
        options=model_list,
    )

    ref_mod_value = widgets.FloatText(
                description='S/m',
                value=1e-3,
    )

    ref_mod = widgets.VBox([
        Label('Reference conductivity'),
        widgets.VBox([ref_type, widgets.VBox([])])
    ])

    start_type = widgets.RadioButtons(
        options=['Model', 'Value'],
        value='Value',
        disabled=False
    )

    start_type.observe(update_start)

    start_mod_value = widgets.FloatText(
                description='S/m',
                value=1e-3,
    )

    start_mod_list = widgets.Dropdown(
        description='3D Model',
        options=model_list,
    )

    start_mod = widgets.VBox([Label('Starting conductivity'), widgets.VBox([start_type, widgets.VBox([start_mod_value])])])

    susc_type = widgets.RadioButtons(
        options=['None', 'Model', 'Value'],
        value='None',
        disabled=False
    )

    susc_type.observe(update_susc)

    susc_mod_value = widgets.FloatText(
                description='S/m',
                value=1e-3,
    )

    susc_mod_list = widgets.Dropdown(
        description='3D Model',
        options=model_list,
    )

    susc_mod = widgets.VBox([Label('Susceptibility model'), widgets.VBox([susc_type])])

    hz_min = widgets.FloatText(
        value=10.,
        description='Smallest cell (m):',
        style={'description_width': 'initial'}
    )

    hz_expansion = widgets.FloatText(
        value=1.05,
        description='Expansion factor:',
        style={'description_width': 'initial'}
    )

    n_cells = widgets.FloatText(
        value=20.,
        description='Number of cells:',
        style={'description_width': 'initial'}
    )

    data_count = Label(
        f"Max depth: {(hz_min.value * hz_expansion.value ** np.arange(n_cells.value)).sum():.2f} m"
    )

    def update_hz_count(_):

        data_count.value = (
            f"Max depth: {(hz_min.value * hz_expansion.value ** np.arange(n_cells.value)).sum():.2f} m"
        )
        write.button_style = 'warning'
        invert.button_style = 'danger'

    n_cells.observe(update_hz_count)
    hz_expansion.observe(update_hz_count)
    hz_min.observe(update_hz_count)

    hz_panel = VBox([hz_min, hz_expansion, n_cells, data_count])

    lower_bound = widgets.FloatText(
        value=1e-5,
        description='Lower bound (S/m)',
        style={'description_width': 'initial'}
    )

    upper_bound = widgets.FloatText(
        value=1e+2,
        description='Upper bound (S/m)',
        style={'description_width': 'initial'}
    )

    bound_panel = VBox([lower_bound, upper_bound])

    norms = widgets.Text(
        value='2, 2, 2, 2',
        description='Norms (m, x, y, z)',
        disabled=False,
        style={'description_width': 'initial'}
    )

    ignore_values = widgets.Text(
        value='<0',
        tooltip='Dummy value',
    )

    inversion_options = {
        "output name": out_group,
        "reference model": ref_mod,
        "starting model": start_mod,
        "upper/lower bounds": bound_panel,
        "mesh 1D": hz_panel,
        "uncertainties": uncert_panel,
        "target misfit": chi_factor,
        "ignore values (<0 = no negatives)": ignore_values,
        "max iterations": max_iteration,
        "norms": norms
    }

    if em_system_specs[system.value]['type'] == 'frequency':
        inversion_options['susceptibity model'] = susc_mod

    option_choices = widgets.Dropdown(
            options=list(inversion_options.keys()),
            value=list(inversion_options.keys())[0],
            disabled=False
    )

    def inv_option_change(_):
        inversion_panel.children[1].children = [option_choices, inversion_options[option_choices.value]]

    option_choices.observe(inv_option_change)
    inversion_panel = widgets.VBox([
        widgets.HBox([widgets.Label("Inversion Options")]),
        widgets.HBox([option_choices, inversion_options[option_choices.value]], )  #layout=widgets.Layout(height='500px')
    ], layout=Layout(width="100%"))

    # Trigger all observers
    if object_name is not None and object_name in names:
        objects.value = object_name
    else:
        object_observer("")

    return widgets.VBox([
        HBox([
            VBox([Label("EM survey"), objects, system, components]),
            VBox([Label("Parameters"), object_fields_panel])
        ]),
        data_panel, inversion_panel, write, invert
    ])