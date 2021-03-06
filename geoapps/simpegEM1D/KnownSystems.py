import numpy as np
import properties
import json


class KnownSystems(properties.HasProperties):
    """Simple Class for KnownSystems."""

    BaseFrequency = properties.Float("WaveFormCurrent",)

    WaveFormCurrent = properties.Array("WaveFormCurrent", dtype=float, shape=("*",))

    WaveFormTime = properties.Array("WaveFormTime", dtype=float, shape=("*",))

    t_peak = properties.Float("Peak waveform time",)

    t0 = properties.Float("Zero waveform time",)

    tref = properties.Float("Reference time for current off",)

    ModellingLoopRadius = properties.Float("ModellingLoopRadius",)

    WindowTimeStart = properties.Array("WindowTimeStart", dtype=float, shape=("*",))

    WindowTimeEnd = properties.Array("WindowTimeEnd", dtype=float, shape=("*",))

    def __init__(self, name):
        self.name = name

    def save(self, fname=None):
        data = self.serialize()
        if fname is None:
            fname = self.name
        with open(fname + ".json", "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False)


def vtem_plus():
    system = KnownSystems("VTEM-plus-7.3ms-pulse")
    # Soource
    system.NumberOfTurns = 1
    system.PeakCurrent = 1
    system.LoopArea = 1
    system.BaseFrequency = 25
    system.WaveformDigitisingFrequency = 192000
    system.WaveFormTime = np.array(
        [
            0.0,
            0.0014974,
            0.00299479,
            0.00449219,
            0.00598958,
            0.00632813,
            0.00666667,
            0.00700521,
            0.00734375,
        ]
    )
    system.WaveFormCurrent = np.array(
        [
            0.00682522,
            0.68821963,
            0.88968217,
            0.95645264,
            1.0,
            0.84188057,
            0.59605229,
            0.296009,
            0.0,
        ]
    )
    system.t_peak = 0.00598958
    system.t0 = 0.00734375
    system.tref = 0.0
    # Receiver
    system.NumberOfWindows = 45
    system.WindowWeightingScheme = "LinearTaper"
    system.WindowTimeStart = np.array(
        [
            0.0000180,
            0.0000230,
            0.0000290,
            0.0000340,
            0.0000390,
            0.0000450,
            0.0000510,
            0.0000590,
            0.0000680,
            0.0000780,
            0.0000900,
            0.0001030,
            0.0001180,
            0.0001360,
            0.0001560,
            0.0001790,
            0.0002060,
            0.0002360,
            0.0002710,
            0.0003120,
            0.0003580,
            0.0004110,
            0.0004720,
            0.0005430,
            0.0006230,
            0.0007160,
            0.0008230,
            0.0009450,
            0.0010860,
            0.0012470,
            0.0014320,
            0.0016460,
            0.0018910,
            0.0021720,
            0.0024950,
            0.0028650,
            0.0032920,
            0.0037810,
            0.0043410,
            0.0049870,
            0.0057290,
            0.0065810,
            0.0075600,
            0.0086850,
            0.0100851,
        ]
    )
    system.WindowTimeEnd = np.array(
        [
            0.0000230,
            0.0000290,
            0.0000340,
            0.0000390,
            0.0000450,
            0.0000510,
            0.0000590,
            0.0000680,
            0.0000780,
            0.0000900,
            0.0001030,
            0.0001180,
            0.0001360,
            0.0001560,
            0.0001790,
            0.0002060,
            0.0002360,
            0.0002710,
            0.0003120,
            0.0003580,
            0.0004110,
            0.0004720,
            0.0005430,
            0.0006230,
            0.0007160,
            0.0008230,
            0.0009450,
            0.0010860,
            0.0012470,
            0.0014320,
            0.0016460,
            0.0018910,
            0.0021720,
            0.0024950,
            0.0028650,
            0.0032920,
            0.0037810,
            0.0043410,
            0.0049870,
            0.0057290,
            0.0065810,
            0.0075600,
            0.0086850,
            0.0099770,
            0.0113498,
        ]
    )
    system.RxNotes = """
        0.0099770 0.0114580 - real Gate 48 as per VTEM specs
        0.0100851 0.0113498 - symetric altered window to prevent linear taper extending into following half cycle
        0.0099770 0.0112957 = non-symetric altered window to prevent linear taper extending into following half cycle
    """
    # Simulation
    system.ModellingLoopRadius = 13
    system.OutputType = "dB/dt"
    system.SrcType = "CircularLoop"
    system.XOutputScaling = 1e12
    system.YOutputScaling = 1e12
    system.ZOutputScaling = 1e12
    system.SecondaryFieldNormalisation = None
    system.FrequenciesPerDecade = 6
    system.NumberOfAbsiccaInHankelTransformEvaluation = 21
    return system


def skytem_hm():
    system = KnownSystems("SkyTem-HighMoment")
    # Soource
    system.NumberOfTurns = 1
    system.PeakCurrent = 1
    system.LoopArea = 1
    system.BaseFrequency = 25
    system.WaveformDigitisingFrequency = 819200
    system.WaveFormTime = np.array(
        [
            -1.000e-02,
            -8.386e-03,
            -6.380e-03,
            -3.783e-03,
            0.000e00,
            3.960e-07,
            7.782e-07,
            1.212e-06,
            3.440e-06,
            1.981e-05,
            3.619e-05,
            3.664e-05,
            3.719e-05,
            3.798e-05,
            3.997e-05,
            1.000e-02,
        ]
    )
    system.WaveFormCurrent = np.array(
        [
            0.000e00,
            4.568e-01,
            7.526e-01,
            9.204e-01,
            1.000e00,
            9.984e-01,
            9.914e-01,
            9.799e-01,
            9.175e-01,
            4.587e-01,
            7.675e-03,
            3.072e-03,
            8.319e-04,
            1.190e-04,
            0.000e00,
            0.000e00,
        ]
    )
    system.t_peak = 0.000e00
    system.t0 = 3.997e-05
    system.tref = 3.997e-05
    # Receiver
    system.NumberOfWindows = 45
    system.WindowWeightingScheme = "LinearTaper"
    system.WindowTimeStart = np.array(
        [
            7.53900e-05,
            9.63900e-05,
            1.22390e-04,
            1.54390e-04,
            1.96390e-04,
            2.47390e-04,
            3.12390e-04,
            3.94390e-04,
            4.97390e-04,
            6.27390e-04,
            7.90390e-04,
            9.96390e-04,
            1.25539e-03,
            1.58139e-03,
            1.99139e-03,
            2.50839e-03,
            3.15839e-03,
            3.97739e-03,
            5.00839e-03,
            6.30639e-03,
            7.93939e-03,
        ]
    )
    system.WindowTimeEnd = np.array(
        [
            9.60000e-05,
            1.22000e-04,
            1.54000e-04,
            1.96000e-04,
            2.47000e-04,
            3.12000e-04,
            3.94000e-04,
            4.97000e-04,
            6.27000e-04,
            7.90000e-04,
            9.96000e-04,
            1.25500e-03,
            1.58100e-03,
            1.99100e-03,
            2.50800e-03,
            3.15800e-03,
            3.97700e-03,
            5.00800e-03,
            6.30600e-03,
            7.93900e-03,
            9.73900e-03,
        ]
    )
    system.RxNotes = """
        Rx Coils 1st order at 300Khz
        Rx Electronics 2nd order at 450Khz
    """
    system.CutOffFrequency = np.array([300000, 450000])
    system.Order = np.array([1, 2], dtype=int)

    # Simulation
    system.ModellingLoopRadius = 9.9975
    system.OutputType = "dB/dt"
    system.SrcType = "CircularLoop"
    system.XOutputScaling = 1
    system.YOutputScaling = 1
    system.ZOutputScaling = 1
    system.SecondaryFieldNormalisation = None
    system.FrequenciesPerDecade = 5
    system.NumberOfAbsiccaInHankelTransformEvaluation = 21
    return system


def skytem_lm():
    system = KnownSystems("SkyTem-LowMoment")
    # Soource
    system.NumberOfTurns = 1
    system.PeakCurrent = 1
    system.LoopArea = 1
    system.BaseFrequency = 222.22222222222222222
    system.WaveformDigitisingFrequency = 3640888.888888889
    system.WaveFormTime = np.array(
        [
            -1.000e-03,
            -9.146e-04,
            -7.879e-04,
            -5.964e-04,
            0.000e00,
            4.629e-07,
            8.751e-07,
            1.354e-06,
            2.540e-06,
            3.972e-06,
            5.404e-06,
            5.721e-06,
            6.113e-06,
            6.663e-06,
            8.068e-06,
            1.250e-03,
        ]
    )
    system.WaveFormCurrent = np.array(
        [
            0.000e00,
            6.264e-01,
            9.132e-01,
            9.905e-01,
            1.000e00,
            9.891e-01,
            9.426e-01,
            8.545e-01,
            6.053e-01,
            3.030e-01,
            4.077e-02,
            1.632e-02,
            4.419e-03,
            6.323e-04,
            0.000e00,
            0.000e00,
        ]
    )
    system.t_peak = 0.000e00
    system.t0 = 8.068e-06
    system.tref = 8.068e-06
    # Receiver
    system.NumberOfWindows = 45
    system.WindowWeightingScheme = "LinearTaper"
    system.WindowTimeStart = np.array(
        [
            0.00001900,
            0.00002400,
            0.00003100,
            0.00003900,
            0.00004900,
            0.00006200,
            0.00007800,
            0.00009900,
            0.00012500,
            0.00015700,
            0.00019900,
            0.00025000,
            0.00031500,
            0.00039700,
            0.00050000,
            0.00063000,
            0.00079300,
            0.00099900,
        ]
    )
    system.WindowTimeEnd = np.array(
        [
            0.00001539,
            0.00001939,
            0.00002439,
            0.00003139,
            0.00003939,
            0.00004939,
            0.00006239,
            0.00007839,
            0.00009939,
            0.00012539,
            0.00015739,
            0.00019939,
            0.00025039,
            0.00031539,
            0.00039739,
            0.00050039,
            0.00063039,
            0.00079339,
        ]
    )

    system.RxNotes = """
        Rx Coils 1st order at 300Khz
        Rx Electronics 2nd order at 450Khz
    """
    system.CutOffFrequency = np.array([300000, 450000])
    system.Order = np.array([1, 2], dtype=int)

    # Simulation
    system.ModellingLoopRadius = 9.9975
    system.OutputType = "dB/dt"
    system.SrcType = "CircularLoop"
    system.XOutputScaling = 1
    system.YOutputScaling = 1
    system.ZOutputScaling = 1
    system.SecondaryFieldNormalisation = None
    system.FrequenciesPerDecade = 5
    system.NumberOfAbsiccaInHankelTransformEvaluation = 21
    return system


def geotem():
    system = KnownSystems("Geotem")
    # Soource
    system.NumberOfTurns = 1
    system.PeakCurrent = 1
    system.LoopArea = 1
    system.BaseFrequency = 25
    system.WaveformDigitisingFrequency = 1638400
    system.WaveFormTime = np.array(
        [
            -0.00410800,
            -0.00397962,
            -0.00385125,
            -0.00372287,
            -0.00359450,
            -0.00346612,
            -0.00333775,
            -0.00320937,
            -0.00308100,
            -0.00295262,
            -0.00282425,
            -0.00269587,
            -0.00256750,
            -0.00243912,
            -0.00231075,
            -0.00218237,
            -0.00205400,
            -0.00192562,
            -0.00179725,
            -0.00166887,
            -0.00154050,
            -0.00141212,
            -0.00128375,
            -0.00115537,
            -0.00102700,
            -0.00089863,
            -0.00077025,
            -0.00064188,
            -0.00051350,
            -0.00038513,
            -0.00025675,
            -0.00012838,
            0.00000000,
        ]
    )
    system.WaveFormCurrent = np.array(
        [
            0.00000000,
            0.09801714,
            0.19509032,
            0.29028468,
            0.38268343,
            0.47139674,
            0.55557023,
            0.63439328,
            0.70710678,
            0.77301045,
            0.83146961,
            0.88192126,
            0.92387953,
            0.95694034,
            0.98078528,
            0.99518473,
            1.00000000,
            0.99518473,
            0.98078528,
            0.95694034,
            0.92387953,
            0.88192126,
            0.83146961,
            0.77301045,
            0.70710678,
            0.63439328,
            0.55557023,
            0.47139674,
            0.38268343,
            0.29028468,
            0.19509032,
            0.09801714,
            0.00000000,
        ]
    )
    system.t_peak = -0.002054
    system.t0 = 0.000e00
    system.tref = 0.000e00
    # Receiver
    system.NumberOfWindows = 16
    system.WindowWeightingScheme = "Boxcar"
    system.WindowTimeStart = np.array(
        [
            0.00027400,
            0.00043100,
            0.00058700,
            0.00074250,
            0.00105550,
            0.00136750,
            0.00183650,
            0.00230550,
            0.00293050,
            0.00371200,
            0.00464900,
            0.00574300,
            0.00699250,
            0.00855550,
            0.01043100,
            0.01293050,
        ]
    )
    system.WindowTimeEnd = np.array(
        [
            0.00043000,
            0.00058700,
            0.00074300,
            0.00105550,
            0.00136850,
            0.00183650,
            0.00230550,
            0.00293050,
            0.00371150,
            0.00465000,
            0.00574300,
            0.00699300,
            0.00855550,
            0.01043050,
            0.01293100,
            0.01574350,
        ]
    )

    system.RxNotes = """ """

    # Simulation
    system.OutputType = "dB/dt"
    system.SrcType = "VMD"
    system.XOutputScaling = 1
    system.YOutputScaling = 1
    system.ZOutputScaling = 1
    system.SecondaryFieldNormalisation = None
    system.FrequenciesPerDecade = 6
    system.NumberOfAbsiccaInHankelTransformEvaluation = 21

    # Geometry
    system.TXRX_DX = -120
    system.TXRX_DZ = -45

    return system


# Have not used yet
def tempest():
    system = KnownSystems("Tempest")
    # Soource
    system.NumberOfTurns = 1
    system.PeakCurrent = 0.5
    system.LoopArea = 1
    system.BaseFrequency = 25
    system.WaveformDigitisingFrequency = 1200000
    system.WaveFormTime = np.array(
        [
            -0.0200000000000,
            -0.0199933333333,
            -0.0000066666667,
            0.0000000000000,
            0.0000066666667,
            0.0199933333333,
            0.0200000000000,
        ]
    )
    system.WaveFormCurrent = np.array([0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0,])

    system.t_peak = -0.002054
    system.t0 = 0.000e00
    system.tref = 0.000e00
    # Receiver
    system.NumberOfWindows = 15
    system.WindowWeightingScheme = "Boxcar"
    system.WindowTimeStart = np.array(
        [
            0.0000066667,
            0.0000333333,
            0.0000600000,
            0.0000866667,
            0.0001400000,
            0.0002200000,
            0.0003533333,
            0.0005666667,
            0.0008866667,
            0.0013666667,
            0.0021133333,
            0.0032866667,
            0.0051266667,
            0.0080066667,
            0.0124066667,
        ]
    )
    system.WindowTimeEnd = np.array(
        [
            0.0000200000,
            0.0000466667,
            0.0000733333,
            0.0001266667,
            0.0002066667,
            0.0003400000,
            0.0005533333,
            0.0008733333,
            0.0013533333,
            0.0021000000,
            0.0032733333,
            0.0051133333,
            0.0079933333,
            0.0123933333,
            0.0199933333,
        ]
    )

    system.RxNotes = """ """

    # Simulation
    system.OutputType = "B"
    system.SrcType = "VMD"
    system.XOutputScaling = 1e15
    system.YOutputScaling = 1e15
    system.ZOutputScaling = 1e15
    system.SecondaryFieldNormalisation = None
    system.FrequenciesPerDecade = 6
    system.NumberOfAbsiccaInHankelTransformEvaluation = 21

    return system
