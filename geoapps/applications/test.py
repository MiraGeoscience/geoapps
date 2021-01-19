from geoapps.processing import ContourValues

from geoapps.plotting import ScatterPlots
from geoapps.processing import Clustering

app = Clustering()
app.h5file = (
    r"C:\Users\dominiquef\Downloads\Geochem_DesurveyedAssay_Cluster_causesErrors.geoh5"
)
app.objects.value = "Renst_DH_Assays_DeSurveyedMiddle"
obj, _ = app.get_selected_entities()
app.data.value = [name for name in app.data.options if name in obj.get_data_list()]
app.widget


# from geoapps.processing import ContourValues
#
# app = ContourValues()
# app.widget
