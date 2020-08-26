import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0

from .. import Utils
from .. import Problem
from .. import Props
import multiprocessing
import properties
from ..Utils import mkvc, matutils, sdiag
from . import BaseMag as MAG
from .MagAnalytics import spheremodel, CongruousMagBC
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.sparse import csr_matrix as csr
import os


class MagneticIntegral(Problem.LinearProblem):

    chi, chiMap, chiDeriv = Props.Invertible(
        "Magnetic Susceptibility (SI)", default=1.0
    )

    forwardOnly = False  # If false, matrix is store to memory (watch your RAM)
    actInd = None  #: Active cell indices provided
    M = None  #: Magnetization matrix provided, otherwise all induced
    magType = "H0"
    verbose = True  # Don't display progress on screen
    W = None
    gtgdiag = None
    n_cpu = None
    parallelized = True
    max_chunk_size = None
    chunk_by_rows = False

    coordinate_system = properties.StringChoice(
        "Type of coordinate system we are regularizing in",
        choices=["cartesian", "spherical"],
        default="cartesian",
    )
    Jpath = "./sensitivity.zarr"
    maxRAM = 1  # Maximum memory usage

    modelType = properties.StringChoice(
        "Type of magnetization model",
        choices=["susceptibility", "vector", "amplitude"],
        default="susceptibility",
    )

    def __init__(self, mesh, **kwargs):

        assert mesh.dim == 3, "Integral formulation only available for 3D mesh"
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        if self.modelType == "vector":
            self.magType = "full"

        # Find non-zero cells
        if getattr(self, "actInd", None) is not None:
            if self.actInd.dtype == "bool":
                inds = (
                    np.asarray(
                        [inds for inds, elem in enumerate(self.actInd, 1) if elem],
                        dtype=int,
                    )
                    - 1
                )
            else:
                inds = self.actInd

        else:

            inds = np.asarray(range(self.mesh.nC))

        self.nC = len(inds)

        # Create active cell projector
        P = csr(
            (np.ones(self.nC), (inds, range(self.nC))), shape=(self.mesh.nC, self.nC)
        )

        # Create vectors of nodal location
        # (lower and upper coners for each cell)
        # if isinstance(self.mesh, Mesh.TreeMesh):
        # Get upper and lower corners of each cell
        bsw = self.mesh.gridCC - self.mesh.h_gridded / 2.0
        tne = self.mesh.gridCC + self.mesh.h_gridded / 2.0

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]

        self.Yn = P.T * np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = P.T * np.c_[mkvc(xn1), mkvc(xn2)]

        if self.mesh.dim > 2:
            zn1, zn2 = bsw[:, 2], tne[:, 2]
            self.Zn = P.T * np.c_[mkvc(zn1), mkvc(zn2)]

        # else:

        #     xn = self.mesh.vectorNx
        #     yn = self.mesh.vectorNy
        #     zn = self.mesh.vectorNz

        #     yn2, xn2, zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
        #     yn1, xn1, zn1 = np.meshgrid(yn[:-1], xn[:-1], zn[:-1])

        # If equivalent source, use semi-infite prism
        # if self.equiSourceLayer:
        #     zn1 -= 1000.

    def fields(self, m):

        if self.coordinate_system == "cartesian":
            m = self.chiMap * m
        else:
            m = self.chiMap * (
                matutils.atp2xyz(m.reshape((int(len(m) / 3), 3), order="F"))
            )

        if self.forwardOnly:
            # Compute the linear operation without forming the full dense F
            return np.array(
                self.Intrgl_Fwr_Op(m=m, magType=self.magType), dtype="float"
            )

        # else:

        if getattr(self, "_Mxyz", None) is not None:

            vec = dask.delayed(csr.dot)(self.Mxyz, m)
            M = da.from_delayed(vec, dtype=float, shape=[m.shape[0]])
            fields = da.dot(self.G, M)

        else:

            fields = da.dot(self.G, m.astype(np.float32))

        if self.modelType == "amplitude":

            fields = self.calcAmpData(fields)

        return fields

    def calcAmpData(self, Bxyz):
        """
            Compute amplitude of the field
        """

        amplitude = da.sum(Bxyz.reshape((3, self.nD), order="F") ** 2.0, axis=0) ** 0.5

        return amplitude

    @property
    def G(self):
        if not self.ispaired:
            raise Exception("Need to pair!")

        if getattr(self, "_G", None) is None:

            self._G = self.Intrgl_Fwr_Op(magType=self.magType)

        return self._G

    @property
    def nD(self):
        """
            Number of data
        """
        self._nD = self.survey.srcField.rxList[0].locs.shape[0]

        return self._nD

    @property
    def ProjTMI(self):
        if not self.ispaired:
            raise Exception("Need to pair!")

        if getattr(self, "_ProjTMI", None) is None:

            # Convert Bdecination from north to cartesian
            self._ProjTMI = Utils.matutils.dipazm_2_xyz(
                self.survey.srcField.param[1], self.survey.srcField.param[2]
            )

        return self._ProjTMI

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """
        dmudm = self.chiMap.deriv(m)
        self._dSdm = None
        self._dfdm = None
        self.model = m
        if (self.gtgdiag is None) and (self.modelType != "amplitude"):

            if W is None:
                W = np.ones(self.G.shape[1])

            self.gtgdiag = np.array(
                da.sum(da.power(W[:, None].astype(np.float32) * self.G, 2), axis=0)
            )

        if self.coordinate_system == "cartesian":
            if self.modelType == "amplitude":
                return np.sum(
                    (self.dfdm * sdiag(mkvc(self.gtgdiag) ** 0.5) * dmudm).power(2.0),
                    axis=0,
                )
            else:
                return mkvc(
                    np.sum(
                        (sdiag(mkvc(self.gtgdiag) ** 0.5) * dmudm).power(2.0), axis=0
                    )
                )

        else:  # spherical
            if self.modelType == "amplitude":
                return mkvc(
                    np.sum(
                        (
                            (self.dfdm)
                            * sdiag(mkvc(self.gtgdiag) ** 0.5)
                            * (self.dSdm * dmudm)
                        ).power(2.0),
                        axis=0,
                    )
                )
            else:

                # Japprox = sdiag(mkvc(self.gtgdiag)**0.5*dmudm) * (self.dSdm * dmudm)
                return mkvc(
                    np.sum(
                        (sdiag(mkvc(self.gtgdiag) ** 0.5) * self.dSdm * dmudm).power(2),
                        axis=0,
                    )
                )

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """

        if self.coordinate_system == "cartesian":
            dmudm = self.chiMap.deriv(m)
        else:  # spherical
            dmudm = self.dSdm * self.chiMap.deriv(m)

        if self.modelType == "amplitude":
            return self.dfdm * da.dot(self.G, dmudm)
        else:

            prod = dask.delayed(csr.dot)(self.G, dmudm)
            return da.from_delayed(
                prod, dtype=float, shape=(self.G.shape[0], dmudm.shape[1])
            )

    def Jvec(self, m, v, f=None):

        if self.coordinate_system == "cartesian":
            dmudm = self.chiMap.deriv(m)
        else:
            dmudm = self.dSdm * self.chiMap.deriv(m)

        if getattr(self, "_Mxyz", None) is not None:

            # dmudm_v = dask.delayed(csr.dot)(dmudm, v)
            # vec = dask.delayed(csr.dot)(self.Mxyz, dmudm_v)
            M_dmudm_v = da.from_array(self.Mxyz * (dmudm * v), chunks=self.G.chunks[1])

            Jvec = da.dot(self.G, M_dmudm_v.astype(np.float32))

        else:

            dmudm_v = da.from_array(dmudm * v, chunks=self.G.chunks[1])

            Jvec = da.dot(self.G, dmudm_v.astype(np.float32))

        if self.modelType == "amplitude":
            dfdm_Jvec = dask.delayed(csr.dot)(self.dfdm, Jvec)

            return da.from_delayed(dfdm_Jvec, dtype=float, shape=[self.dfdm.shape[0]])
        else:
            return Jvec

    def Jtvec(self, m, v, f=None):

        if self.coordinate_system == "cartesian":
            dmudm = self.chiMap.deriv(m)
        else:
            dmudm = self.dSdm * self.chiMap.deriv(m)

        if self.modelType == "amplitude":

            dfdm_v = dask.delayed(csr.dot)(v, self.dfdm)

            vec = da.from_delayed(dfdm_v, dtype=float, shape=[self.dfdm.shape[0]])

            if getattr(self, "_Mxyz", None) is not None:

                jtvec = da.dot(vec.astype(np.float32), self.G)

                Jtvec = dask.delayed(csr.dot)(jtvec, self.Mxyz)

            else:
                Jtvec = da.dot(vec.astype(np.float32), self.G)

        else:

            Jtvec = da.dot(v.astype(np.float32), self.G)

        dmudm_v = dask.delayed(csr.dot)(Jtvec, dmudm)

        return da.from_delayed(dmudm_v, dtype=float, shape=[dmudm.shape[1]])

    @property
    def dSdm(self):

        if getattr(self, "_dSdm", None) is None:

            if self.model is None:
                raise Exception("Requires a chi")

            nC = int(len(self.model) / 3)

            m_xyz = self.chiMap * matutils.atp2xyz(
                self.model.reshape((nC, 3), order="F")
            )

            nC = int(m_xyz.shape[0] / 3.0)
            m_atp = matutils.xyz2atp(m_xyz.reshape((nC, 3), order="F"))

            a = m_atp[:nC]
            t = m_atp[nC : 2 * nC]
            p = m_atp[2 * nC :]

            Sx = sp.hstack(
                [
                    sp.diags(np.cos(t) * np.cos(p), 0),
                    sp.diags(-a * np.sin(t) * np.cos(p), 0),
                    sp.diags(-a * np.cos(t) * np.sin(p), 0),
                ]
            )

            Sy = sp.hstack(
                [
                    sp.diags(np.cos(t) * np.sin(p), 0),
                    sp.diags(-a * np.sin(t) * np.sin(p), 0),
                    sp.diags(a * np.cos(t) * np.cos(p), 0),
                ]
            )

            Sz = sp.hstack(
                [sp.diags(np.sin(t), 0), sp.diags(a * np.cos(t), 0), csr((nC, nC))]
            )

            self._dSdm = sp.vstack([Sx, Sy, Sz])

        return self._dSdm

    @property
    def modelMap(self):
        """
            Call for general mapping of the problem
        """
        return self.chiMap

    @property
    def dfdm(self):

        if self.model is None:
            self.model = np.zeros(self.G.shape[1])

        if getattr(self, "_dfdm", None) is None:

            Bxyz = self.Bxyz_a(self.chiMap * self.model)

            # Bx = sp.spdiags(Bxyz[:, 0], 0, self.nD, self.nD)
            # By = sp.spdiags(Bxyz[:, 1], 0, self.nD, self.nD)
            # Bz = sp.spdiags(Bxyz[:, 2], 0, self.nD, self.nD)
            ii = np.kron(np.asarray(range(self.survey.nD), dtype="int"), np.ones(3))
            jj = np.asarray(range(3 * self.survey.nD), dtype="int")
            # (data, (row, col)), shape=(3, 3))
            # P = s
            self._dfdm = csr(
                (mkvc(Bxyz), (ii, jj)), shape=(self.survey.nD, 3 * self.survey.nD)
            )

        return self._dfdm

    def Bxyz_a(self, m):
        """
            Return the normalized B fields
        """

        # Get field data
        if self.coordinate_system == "spherical":
            m = matutils.atp2xyz(m)

        if getattr(self, "_Mxyz", None) is not None:
            Bxyz = da.dot(self.G, (self.Mxyz * m).astype(np.float32))
        else:
            Bxyz = da.dot(self.G, m.astype(np.float32))

        amp = self.calcAmpData(Bxyz.astype(np.float64))
        Bamp = sp.spdiags(1.0 / amp, 0, self.nD, self.nD)

        return Bxyz.reshape((3, self.nD), order="F") * Bamp

    def Intrgl_Fwr_Op(self, m=None, magType="H0"):
        """

        Magnetic forward operator in integral form

        magType  = 'H0' | 'x' | 'y' | 'z'
        components  = 'tmi' | 'x' | 'y' | 'z'

        Return
        _G = Linear forward operator | (forwardOnly)=data

         """
        # if m is not None:
        #     self.model = self.chiMap * m

        # survey = self.survey
        self.rxLoc = self.survey.srcField.rxList[0].locs

        if magType == "H0":
            if getattr(self, "M", None) is None:
                self.M = matutils.dipazm_2_xyz(
                    np.ones(self.nC) * self.survey.srcField.param[1],
                    np.ones(self.nC) * self.survey.srcField.param[2],
                )

            Mx = sdiag(self.M[:, 0] * self.survey.srcField.param[0])
            My = sdiag(self.M[:, 1] * self.survey.srcField.param[0])
            Mz = sdiag(self.M[:, 2] * self.survey.srcField.param[0])

            self.Mxyz = sp.vstack((Mx, My, Mz))

        elif magType == "full":

            self.Mxyz = sp.identity(3 * self.nC) * self.survey.srcField.param[0]
        else:
            raise Exception('magType must be: "H0" or "full"')

            # Loop through all observations and create forward operator (nD-by-self.nC)

        if self.verbose:
            print(
                "Begin forward: M=" + magType + ", Rx type= %s" % self.survey.components
            )

        # Switch to determine if the process has to be run in parallel
        job = Forward(
            rxLoc=self.rxLoc,
            Xn=self.Xn,
            Yn=self.Yn,
            Zn=self.Zn,
            n_cpu=self.n_cpu,
            forwardOnly=self.forwardOnly,
            model=m,
            components=self.survey.components,
            Mxyz=self.Mxyz,
            P=self.ProjTMI,
            parallelized=self.parallelized,
            verbose=self.verbose,
            Jpath=self.Jpath,
            maxRAM=self.maxRAM,
            max_chunk_size=self.max_chunk_size,
            chunk_by_rows=self.chunk_by_rows,
        )

        G = job.calculate()

        return G


class Forward:

    progressIndex = -1
    parallelized = True
    rxLoc = None
    Xn, Yn, Zn = None, None, None
    n_cpu = None
    forwardOnly = False
    components = ["tmi"]
    model = None
    Mxyz = None
    P = None
    verbose = True
    maxRAM = 1
    chunk_by_rows = False

    max_chunk_size = None
    Jpath = "./sensitivity.zarr"

    def __init__(self, **kwargs):
        super().__init__()
        Utils.setKwargs(self, **kwargs)

    def calculate(self):
        self.nD = self.rxLoc.shape[0]
        self.nC = self.Mxyz.shape[1]

        if self.n_cpu is None:
            self.n_cpu = int(multiprocessing.cpu_count())

        # Set this early so we can get a better memory estimate for dask chunking
        # if self.components == 'xyz':
        #     nDataComps = 3
        # else:
        nDataComps = len(self.components)

        if self.parallelized:

            row = dask.delayed(self.calcTrow, pure=True)

            makeRows = [row(self.rxLoc[ii, :]) for ii in range(self.nD)]

            buildMat = [
                da.from_delayed(makeRow, dtype=np.float32, shape=(nDataComps, self.nC))
                for makeRow in makeRows
            ]

            stack = da.vstack(buildMat)

            # Auto rechunk
            # To customise memory use set Dask config in calling scripts: dask.config.set({'array.chunk-size': '128MiB'})
            if self.forwardOnly or self.chunk_by_rows:
                label = "DASK: Chunking by rows"
                # Autochunking by rows is faster and more memory efficient for
                # very large problems sensitivty and forward calculations
                target_size = dask.config.get("array.chunk-size").replace("MiB", " MB")
                stack = stack.rechunk({0: "auto", 1: -1})
            elif self.max_chunk_size:
                label = "DASK: Chunking using parameters"
                # Manual chunking is less sensitive to chunk sizes for some problems
                target_size = f"{self.max_chunk_size:.0f} MB"
                nChunks_col = 1
                nChunks_row = 1
                rowChunk = int(np.ceil(stack.shape[0] / nChunks_row))
                colChunk = int(np.ceil(stack.shape[1] / nChunks_col))
                chunk_size = rowChunk * colChunk * 8 * 1e-6  # in Mb

                # Add more chunks until memory falls below target
                while chunk_size >= self.max_chunk_size:

                    if rowChunk > colChunk:
                        nChunks_row += 1
                    else:
                        nChunks_col += 1

                    rowChunk = int(np.ceil(stack.shape[0] / nChunks_row))
                    colChunk = int(np.ceil(stack.shape[1] / nChunks_col))
                    chunk_size = rowChunk * colChunk * 8 * 1e-6  # in Mb

                stack = stack.rechunk((rowChunk, colChunk))
            else:
                label = "DASK: Chunking by columns"
                # Autochunking by columns is faster for Inversions
                target_size = dask.config.get("array.chunk-size").replace("MiB", " MB")
                stack = stack.rechunk({0: -1, 1: "auto"})

            if self.verbose:
                print(label)
                print("Tile size (nD, nC): ", stack.shape)
                #                print('Chunk sizes (nD, nC): ', stack.chunks) # For debugging only
                print(
                    "Number of chunks: %.0f x %.0f = %.0f"
                    % (
                        len(stack.chunks[0]),
                        len(stack.chunks[1]),
                        len(stack.chunks[0]) * len(stack.chunks[1]),
                    )
                )
                print("Target chunk size: %s" % target_size)
                print(
                    "Max chunk size %.0f x %.0f = %.3f MB"
                    % (
                        max(stack.chunks[0]),
                        max(stack.chunks[1]),
                        max(stack.chunks[0]) * max(stack.chunks[1]) * 8 * 1e-6,
                    )
                )
                print(
                    "Min chunk size %.0f x %.0f = %.3f MB"
                    % (
                        min(stack.chunks[0]),
                        min(stack.chunks[1]),
                        min(stack.chunks[0]) * min(stack.chunks[1]) * 8 * 1e-6,
                    )
                )
                print(
                    "Max RAM (GB x %.0f CPU): %.6f"
                    % (
                        self.n_cpu,
                        max(stack.chunks[0])
                        * max(stack.chunks[1])
                        * 8
                        * 1e-9
                        * self.n_cpu,
                    )
                )
                print(
                    "Tile size (GB): %.3f"
                    % (stack.shape[0] * stack.shape[1] * 8 * 1e-9)
                )

            if self.forwardOnly:

                with ProgressBar():
                    print("Forward calculation: ")
                    pred = da.dot(stack, self.model).compute()

                return pred

            else:

                if os.path.exists(self.Jpath):

                    G = da.from_zarr(self.Jpath)

                    if np.all(
                        np.r_[
                            np.any(np.r_[G.chunks[0]] == stack.chunks[0]),
                            np.any(np.r_[G.chunks[1]] == stack.chunks[1]),
                            np.r_[G.shape] == np.r_[stack.shape],
                        ]
                    ):
                        # Check that loaded G matches supplied data and mesh
                        print(
                            "Zarr file detected with same shape and chunksize ... re-loading"
                        )

                        return G
                    else:

                        print(
                            "Zarr file detected with wrong shape and chunksize ... over-writing"
                        )

                with ProgressBar():
                    print("Saving G to zarr: " + self.Jpath)
                    G = da.to_zarr(
                        stack,
                        self.Jpath,
                        compute=True,
                        return_stored=True,
                        overwrite=True,
                    )

        else:

            result = []
            for ii in range(self.nD):

                if self.forwardOnly:
                    result += [
                        np.c_[np.dot(self.calcTrow(self.rxLoc[ii, :]), self.model)]
                    ]
                else:
                    result += [self.calcTrow(self.rxLoc[ii, :])]
                self.progress(ii, self.nD)

            G = np.vstack(result)

        return G

    def calcTrow(self, xyzLoc):
        """
            Load in the active nodes of a tensor mesh and computes the magnetic
            forward relation between a cuboid and a given observation
            location outside the Earth [obsx, obsy, obsz]

            INPUT:
            xyzLoc:  [obsx, obsy, obsz] nC x 3 Array

            OUTPUT:
            Tx = [Txx Txy Txz]
            Ty = [Tyx Tyy Tyz]
            Tz = [Tzx Tzy Tzz]

        """

        rows = calcRow(
            self.Xn, self.Yn, self.Zn, xyzLoc, self.P, components=self.components
        )

        return rows * self.Mxyz

    def progress(self, ind, total):
        """
        progress(ind,prog,final)

        Function measuring the progress of a process and print to screen the %.
        Useful to estimate the remaining runtime of a large problem.

        Created on Dec, 20th 2015

        @author: dominiquef
        """
        arg = np.floor(ind / total * 10.0)
        if arg > self.progressIndex:

            if self.verbose:
                print("Done " + str(arg * 10) + " %")
            self.progressIndex = arg


class Problem3D_DiffSecondary(Problem.BaseProblem):
    """
        Secondary field approach using differential equations!
    """

    surveyPair = MAG.BaseMagSurvey
    modelPair = MAG.BaseMagMap

    mu, muMap, muDeriv = Props.Invertible("Magnetic Permeability (H/m)", default=mu_0)

    mui, muiMap, muiDeriv = Props.Invertible("Inverse Magnetic Permeability (m/H)")

    Props.Reciprocal(mu, mui)

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        Pbc, Pin, self._Pout = self.mesh.getBCProjWF("neumann", discretization="CC")

        Dface = self.mesh.faceDiv
        Mc = sdiag(self.mesh.vol)
        self._Div = Mc * Dface * Pin.T * Pin

    @property
    def MfMuI(self):
        return self._MfMuI

    @property
    def MfMui(self):
        return self._MfMui

    @property
    def MfMu0(self):
        return self._MfMu0

    def makeMassMatrices(self, m):
        mu = self.muMap * m
        self._MfMui = self.mesh.getFaceInnerProduct(1.0 / mu) / self.mesh.dim
        # self._MfMui = self.mesh.getFaceInnerProduct(1./mu)
        # TODO: this will break if tensor mu
        self._MfMuI = sdiag(1.0 / self._MfMui.diagonal())
        self._MfMu0 = self.mesh.getFaceInnerProduct(1.0 / mu_0) / self.mesh.dim
        # self._MfMu0 = self.mesh.getFaceInnerProduct(1/mu_0)

    @Utils.requires("survey")
    def getB0(self):
        b0 = self.survey.B0
        B0 = np.r_[
            b0[0] * np.ones(self.mesh.nFx),
            b0[1] * np.ones(self.mesh.nFy),
            b0[2] * np.ones(self.mesh.nFz),
        ]
        return B0

    def getRHS(self, m):
        r"""

        .. math ::

            \mathbf{rhs} = \Div(\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0 - \Div\mathbf{B}_0+\diag(v)\mathbf{D} \mathbf{P}_{out}^T \mathbf{B}_{sBC}

        """
        B0 = self.getB0()
        Dface = self.mesh.faceDiv
        # Mc = sdiag(self.mesh.vol)

        mu = self.muMap * m
        chi = mu / mu_0 - 1

        # Temporary fix
        Bbc, Bbc_const = CongruousMagBC(self.mesh, self.survey.B0, chi)
        self.Bbc = Bbc
        self.Bbc_const = Bbc_const
        # return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0 +
        # Mc*Dface*self._Pout.T*Bbc
        return self._Div * self.MfMuI * self.MfMu0 * B0 - self._Div * B0

    def getA(self, m):
        r"""
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return self._Div * self.MfMuI * self._Div.T

    def fields(self, m):
        r"""
            Return magnetic potential (u) and flux (B)
            u: defined on the cell center [nC x 1]
            B: defined on the cell center [nG x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        self.makeMassMatrices(m)
        A = self.getA(m)
        rhs = self.getRHS(m)
        m1 = sp.linalg.interface.aslinearoperator(sdiag(1 / A.diagonal()))
        u, info = sp.linalg.bicgstab(A, rhs, tol=1e-6, maxiter=1000, M=m1)
        B0 = self.getB0()
        B = self.MfMuI * self.MfMu0 * B0 - B0 - self.MfMuI * self._Div.T * u

        return {"B": B, "u": u}

    @Utils.timeIt
    def Jvec(self, m, v, u=None):
        """
            Computing Jacobian multiplied by vector

            By setting our problem as

            .. math ::

                \\mathbf{C}(\\mathbf{m}, \\mathbf{u}) = \\mathbf{A}\\mathbf{u} - \\mathbf{rhs} = 0

            And taking derivative w.r.t m

            .. math ::

                \\nabla \\mathbf{C}(\\mathbf{m}, \\mathbf{u}) = \\nabla_m \\mathbf{C}(\\mathbf{m}) \\delta \\mathbf{m} +
                                                             \\nabla_u \\mathbf{C}(\\mathbf{u}) \\delta \\mathbf{u} = 0

                \\frac{\\delta \\mathbf{u}}{\\delta \\mathbf{m}} = - [\\nabla_u \\mathbf{C}(\\mathbf{u})]^{-1}\\nabla_m \\mathbf{C}(\\mathbf{m})

            With some linear algebra we can have

            .. math ::

                \\nabla_u \\mathbf{C}(\\mathbf{u}) = \\mathbf{A}

                \\nabla_m \\mathbf{C}(\\mathbf{m}) =
                \\frac{\\partial \\mathbf{A}}{\\partial \\mathbf{m}}(\\mathbf{m})\\mathbf{u} - \\frac{\\partial \\mathbf{rhs}(\\mathbf{m})}{\\partial \\mathbf{m}}

            .. math ::

                \\frac{\\partial \\mathbf{A}}{\\partial \\mathbf{m}}(\\mathbf{m})\\mathbf{u} =
                \\frac{\\partial \\mathbf{\\mu}}{\\partial \\mathbf{m}} \\left[\\Div \\diag (\\Div^T \\mathbf{u}) \\dMfMuI \\right]

                \\dMfMuI = \\diag(\\MfMui)^{-1}_{vec} \\mathbf{Av}_{F2CC}^T\\diag(\\mathbf{v})\\diag(\\frac{1}{\\mu^2})

                \\frac{\\partial \\mathbf{rhs}(\\mathbf{m})}{\\partial \\mathbf{m}} =  \\frac{\\partial \\mathbf{\\mu}}{\\partial \\mathbf{m}} \\left[
                \\Div \\diag(\\M^f_{\\mu_{0}^{-1}}\\mathbf{B}_0) \\dMfMuI \\right] - \\diag(\\mathbf{v})\\mathbf{D} \\mathbf{P}_{out}^T\\frac{\\partial B_{sBC}}{\\partial \\mathbf{m}}

            In the end,

            .. math ::

                \\frac{\\delta \\mathbf{u}}{\\delta \\mathbf{m}} =
                - [ \\mathbf{A} ]^{-1}\\left[ \\frac{\\partial \\mathbf{A}}{\\partial \\mathbf{m}}(\\mathbf{m})\\mathbf{u}
                - \\frac{\\partial \\mathbf{rhs}(\\mathbf{m})}{\\partial \\mathbf{m}} \\right]

            A little tricky point here is we are not interested in potential (u), but interested in magnetic flux (B).
            Thus, we need sensitivity for B. Now we take derivative of B w.r.t m and have

            .. math ::

                \\frac{\\delta \\mathbf{B}} {\\delta \\mathbf{m}} = \\frac{\\partial \\mathbf{\\mu} } {\\partial \\mathbf{m} }
                \\left[
                \\diag(\\M^f_{\\mu_{0}^{-1} } \\mathbf{B}_0) \\dMfMuI  \\
                 -  \\diag (\\Div^T\\mathbf{u})\\dMfMuI
                \\right ]

                 -  (\\MfMui)^{-1}\\Div^T\\frac{\\delta\\mathbf{u}}{\\delta \\mathbf{m}}

            Finally we evaluate the above, but we should remember that

            .. note ::

                We only want to evalute

                .. math ::

                    \\mathbf{J}\\mathbf{v} = \\frac{\\delta \\mathbf{P}\\mathbf{B}} {\\delta \\mathbf{m}}\\mathbf{v}

                Since forming sensitivity matrix is very expensive in that this monster is "big" and "dense" matrix!!


        """
        if u is None:
            u = self.fields(m)

        B, u = u["B"], u["u"]
        mu = self.muMap * (m)
        dmudm = self.muDeriv
        # dchidmu = sdiag(1 / mu_0 * np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1 / self.MfMui.diagonal()
        dMfMuI = sdiag(MfMuIvec ** 2) * self.mesh.aveF2CC.T * sdiag(vol * 1.0 / mu ** 2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        dCdm_A = Div * (sdiag(Div.T * u) * dMfMuI * dmudm)
        dCdm_RHS1 = Div * (sdiag(self.MfMu0 * B0) * dMfMuI)
        # temp1 = (Dface * (self._Pout.T * self.Bbc_const * self.Bbc))
        # dCdm_RHS2v = (sdiag(vol) * temp1) * \
        #    np.inner(vol, dchidmu * dmudm * v)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        dCdm_RHSv = dCdm_RHS1 * (dmudm * v)
        dCdm_v = dCdm_A * v - dCdm_RHSv

        m1 = sp.linalg.interface.aslinearoperator(sdiag(1 / dCdu.diagonal()))
        sol, info = sp.linalg.bicgstab(dCdu, dCdm_v, tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print("Iterative solver did not work well (Jvec)")
            # raise Exception ("Iterative solver did not work well")

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu

        dudm = -sol
        dBdmv = (
            sdiag(self.MfMu0 * B0) * (dMfMuI * (dmudm * v))
            - sdiag(Div.T * u) * (dMfMuI * (dmudm * v))
            - self.MfMuI * (Div.T * (dudm))
        )

        return mkvc(P * dBdmv)

    @Utils.timeIt
    def Jtvec(self, m, v, u=None):
        """
            Computing Jacobian^T multiplied by vector.

        .. math ::

            (\\frac{\\delta \\mathbf{P}\\mathbf{B}} {\\delta \\mathbf{m}})^{T} = \\left[ \\mathbf{P}_{deriv}\\frac{\\partial \\mathbf{\\mu} } {\\partial \\mathbf{m} }
            \\left[
            \\diag(\\M^f_{\\mu_{0}^{-1} } \\mathbf{B}_0) \\dMfMuI  \\
             -  \\diag (\\Div^T\\mathbf{u})\\dMfMuI
            \\right ]\\right]^{T}

             -  \\left[\\mathbf{P}_{deriv}(\\MfMui)^{-1}\\Div^T\\frac{\\delta\\mathbf{u}}{\\delta \\mathbf{m}} \\right]^{T}

        where

        .. math ::

            \\mathbf{P}_{derv} = \\frac{\\partial \\mathbf{P}}{\\partial\\mathbf{B}}

        .. note ::

            Here we only want to compute

            .. math ::

                \\mathbf{J}^{T}\\mathbf{v} = (\\frac{\\delta \\mathbf{P}\\mathbf{B}} {\\delta \\mathbf{m}})^{T} \\mathbf{v}

        """
        if u is None:
            u = self.fields(m)

        B, u = u["B"], u["u"]
        mu = self.mapping * (m)
        dmudm = self.mapping.deriv(m)
        # dchidmu = sdiag(1 / mu_0 * np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1 / self.MfMui.diagonal()
        dMfMuI = sdiag(MfMuIvec ** 2) * self.mesh.aveF2CC.T * sdiag(vol * 1.0 / mu ** 2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        s = Div * (self.MfMuI.T * (P.T * v))

        m1 = sp.linalg.interface.aslinearoperator(sdiag(1 / (dCdu.T).diagonal()))
        sol, info = sp.linalg.bicgstab(dCdu.T, s, tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print("Iterative solver did not work well (Jtvec)")
            # raise Exception ("Iterative solver did not work well")

        # dCdm_A = Div * ( sdiag( Div.T * u )* dMfMuI *dmudm  )
        # dCdm_Atsol = ( dMfMuI.T*( sdiag( Div.T * u ) * (Div.T * dmudm)) ) * sol
        dCdm_Atsol = (dmudm.T * dMfMuI.T * (sdiag(Div.T * u) * Div.T)) * sol

        # dCdm_RHS1 = Div * (sdiag( self.MfMu0*B0  ) * dMfMuI)
        # dCdm_RHS1tsol = (dMfMuI.T*( sdiag( self.MfMu0*B0  ) ) * Div.T * dmudm) * sol
        dCdm_RHS1tsol = (dmudm.T * dMfMuI.T * (sdiag(self.MfMu0 * B0)) * Div.T) * sol

        # temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        # temp1sol = (Dface.T * (sdiag(vol) * sol))
        # temp2 = self.Bbc_const * (self._Pout.T * self.Bbc).T
        # dCdm_RHS2v  = (sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)
        # dCdm_RHS2tsol = (dmudm.T * dchidmu.T * vol) * np.inner(temp2, temp1sol)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v

        # temporary fix
        # dCdm_RHStsol = dCdm_RHS1tsol - dCdm_RHS2tsol
        dCdm_RHStsol = dCdm_RHS1tsol

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        # dCdm_v = dCdm_A*v - dCdm_RHSv

        Ctv = dCdm_Atsol - dCdm_RHStsol

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu
        # dPBdm^T*v = Atemp^T*P^T*v - Btemp^T*P^T*v - Ctv

        Atemp = sdiag(self.MfMu0 * B0) * (dMfMuI * (dmudm))
        Btemp = sdiag(Div.T * u) * (dMfMuI * (dmudm))
        Jtv = Atemp.T * (P.T * v) - Btemp.T * (P.T * v) - Ctv

        return mkvc(Jtv)


def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):
    """
        Inversion module for MagneticsDiffSecondary

    """
    from .. import Optimization, Regularization, Parameters, ObjFunction, Inversion

    prob = MagneticsDiffSecondary(mesh, model)

    miter = kwargs.get("maxIter", 10)

    if prob.ispaired:
        prob.unpair()
    if data.ispaired:
        data.unpair()
    prob.pair(data)

    # Create an optimization program
    opt = Optimization.InexactGaussNewton(maxIter=miter)
    opt.bfgsH0 = Solver(sp.identity(model.nP), flag="D")
    # Create a regularization program
    reg = Regularization.Tikhonov(model)
    # Create an objective function
    beta = Parameters.BetaSchedule(beta0=1e0)
    obj = ObjFunction.BaseObjFunction(data, reg, beta=beta)
    # Create an inversion object
    inv = Inversion.BaseInversion(obj, opt)

    return inv, reg


def calcRow(
    Xn,
    Yn,
    Zn,
    rxlocation,
    P,
    components=["bxx", "bxy", "bxz", "byy", "byz", "bzz", "bx", "by", "bz"],
):
    """
    calcRow
    Takes in the lower SW and upper NE nodes of a tensor mesh,
    observation location rxLoc[obsx, obsy, obsz] and computes the
    magnetic tensor for the integral of a each prisms

    INPUT:
    Xn, Yn, Zn: Node location matrix for the lower and upper most corners of
                all cells in the mesh shape[nC,2]
    OUTPUT:

    """
    eps = 1e-8  # add a small value to the locations to avoid /0
    # number of cells in mesh
    nC = Xn.shape[0]

    # comp. pos. differences for tne, bsw nodes
    dz2 = Zn[:, 1] - rxlocation[2] + eps
    dz1 = Zn[:, 0] - rxlocation[2] + eps

    dy2 = Yn[:, 1] - rxlocation[1] + eps
    dy1 = Yn[:, 0] - rxlocation[1] + eps

    dx2 = Xn[:, 1] - rxlocation[0] + eps
    dx1 = Xn[:, 0] - rxlocation[0] + eps

    # comp. squared diff
    dx2dx2 = dx2 ** 2.0
    dx1dx1 = dx1 ** 2.0

    dy2dy2 = dy2 ** 2.0
    dy1dy1 = dy1 ** 2.0

    dz2dz2 = dz2 ** 2.0
    dz1dz1 = dz1 ** 2.0

    # 2D radius compent squared of corner nodes
    R1 = dy2dy2 + dx2dx2
    R2 = dy2dy2 + dx1dx1
    R3 = dy1dy1 + dx2dx2
    R4 = dy1dy1 + dx1dx1

    # radius to each cell node
    r1 = np.sqrt(dz2dz2 + R2) + eps
    r2 = np.sqrt(dz2dz2 + R1) + eps
    r3 = np.sqrt(dz1dz1 + R1) + eps
    r4 = np.sqrt(dz1dz1 + R2) + eps
    r5 = np.sqrt(dz2dz2 + R3) + eps
    r6 = np.sqrt(dz2dz2 + R4) + eps
    r7 = np.sqrt(dz1dz1 + R4) + eps
    r8 = np.sqrt(dz1dz1 + R3) + eps

    # compactify argument calculations
    arg1_ = dx1 + dy2 + r1
    arg1 = dy2 + dz2 + r1
    arg2 = dx1 + dz2 + r1
    arg3 = dx1 + r1
    arg4 = dy2 + r1
    arg5 = dz2 + r1

    arg6_ = dx2 + dy2 + r2
    arg6 = dy2 + dz2 + r2
    arg7 = dx2 + dz2 + r2
    arg8 = dx2 + r2
    arg9 = dy2 + r2
    arg10 = dz2 + r2

    arg11_ = dx2 + dy2 + r3
    arg11 = dy2 + dz1 + r3
    arg12 = dx2 + dz1 + r3
    arg13 = dx2 + r3
    arg14 = dy2 + r3
    arg15 = dz1 + r3

    arg16_ = dx1 + dy2 + r4
    arg16 = dy2 + dz1 + r4
    arg17 = dx1 + dz1 + r4
    arg18 = dx1 + r4
    arg19 = dy2 + r4
    arg20 = dz1 + r4

    arg21_ = dx2 + dy1 + r5
    arg21 = dy1 + dz2 + r5
    arg22 = dx2 + dz2 + r5
    arg23 = dx2 + r5
    arg24 = dy1 + r5
    arg25 = dz2 + r5

    arg26_ = dx1 + dy1 + r6
    arg26 = dy1 + dz2 + r6
    arg27 = dx1 + dz2 + r6
    arg28 = dx1 + r6
    arg29 = dy1 + r6
    arg30 = dz2 + r6

    arg31_ = dx1 + dy1 + r7
    arg31 = dy1 + dz1 + r7
    arg32 = dx1 + dz1 + r7
    arg33 = dx1 + r7
    arg34 = dy1 + r7
    arg35 = dz1 + r7

    arg36_ = dx2 + dy1 + r8
    arg36 = dy1 + dz1 + r8
    arg37 = dx2 + dz1 + r8
    arg38 = dx2 + r8
    arg39 = dy1 + r8
    arg40 = dz1 + r8

    rows = []
    bxx, byy = [], []
    for comp in components:
        # m_x vector
        if (comp == "bxx") or ("bzz" in components):
            bxx = np.zeros((1, 3 * nC))

            bxx[0, 0:nC] = 2 * (
                ((dx1 ** 2 - r1 * arg1) / (r1 * arg1 ** 2 + dx1 ** 2 * r1 + eps))
                - ((dx2 ** 2 - r2 * arg6) / (r2 * arg6 ** 2 + dx2 ** 2 * r2 + eps))
                + ((dx2 ** 2 - r3 * arg11) / (r3 * arg11 ** 2 + dx2 ** 2 * r3 + eps))
                - ((dx1 ** 2 - r4 * arg16) / (r4 * arg16 ** 2 + dx1 ** 2 * r4 + eps))
                + ((dx2 ** 2 - r5 * arg21) / (r5 * arg21 ** 2 + dx2 ** 2 * r5 + eps))
                - ((dx1 ** 2 - r6 * arg26) / (r6 * arg26 ** 2 + dx1 ** 2 * r6 + eps))
                + ((dx1 ** 2 - r7 * arg31) / (r7 * arg31 ** 2 + dx1 ** 2 * r7 + eps))
                - ((dx2 ** 2 - r8 * arg36) / (r8 * arg36 ** 2 + dx2 ** 2 * r8 + eps))
            )

            bxx[0, nC : 2 * nC] = (
                dx2 / (r5 * arg25 + eps)
                - dx2 / (r2 * arg10 + eps)
                + dx2 / (r3 * arg15 + eps)
                - dx2 / (r8 * arg40 + eps)
                + dx1 / (r1 * arg5 + eps)
                - dx1 / (r6 * arg30 + eps)
                + dx1 / (r7 * arg35 + eps)
                - dx1 / (r4 * arg20 + eps)
            )

            bxx[0, 2 * nC :] = (
                dx1 / (r1 * arg4 + eps)
                - dx2 / (r2 * arg9 + eps)
                + dx2 / (r3 * arg14 + eps)
                - dx1 / (r4 * arg19 + eps)
                + dx2 / (r5 * arg24 + eps)
                - dx1 / (r6 * arg29 + eps)
                + dx1 / (r7 * arg34 + eps)
                - dx2 / (r8 * arg39 + eps)
            )

            bxx /= 4 * np.pi

        if (comp == "byy") or ("bzz" in components):
            # byy
            byy = np.zeros((1, 3 * nC))

            byy[0, 0:nC] = (
                dy2 / (r3 * arg15 + eps)
                - dy2 / (r2 * arg10 + eps)
                + dy1 / (r5 * arg25 + eps)
                - dy1 / (r8 * arg40 + eps)
                + dy2 / (r1 * arg5 + eps)
                - dy2 / (r4 * arg20 + eps)
                + dy1 / (r7 * arg35 + eps)
                - dy1 / (r6 * arg30 + eps)
            )
            byy[0, nC : 2 * nC] = 2 * (
                ((dy2 ** 2 - r1 * arg2) / (r1 * arg2 ** 2 + dy2 ** 2 * r1 + eps))
                - ((dy2 ** 2 - r2 * arg7) / (r2 * arg7 ** 2 + dy2 ** 2 * r2 + eps))
                + ((dy2 ** 2 - r3 * arg12) / (r3 * arg12 ** 2 + dy2 ** 2 * r3 + eps))
                - ((dy2 ** 2 - r4 * arg17) / (r4 * arg17 ** 2 + dy2 ** 2 * r4 + eps))
                + ((dy1 ** 2 - r5 * arg22) / (r5 * arg22 ** 2 + dy1 ** 2 * r5 + eps))
                - ((dy1 ** 2 - r6 * arg27) / (r6 * arg27 ** 2 + dy1 ** 2 * r6 + eps))
                + ((dy1 ** 2 - r7 * arg32) / (r7 * arg32 ** 2 + dy1 ** 2 * r7 + eps))
                - ((dy1 ** 2 - r8 * arg37) / (r8 * arg37 ** 2 + dy1 ** 2 * r8 + eps))
            )
            byy[0, 2 * nC :] = (
                dy2 / (r1 * arg3 + eps)
                - dy2 / (r2 * arg8 + eps)
                + dy2 / (r3 * arg13 + eps)
                - dy2 / (r4 * arg18 + eps)
                + dy1 / (r5 * arg23 + eps)
                - dy1 / (r6 * arg28 + eps)
                + dy1 / (r7 * arg33 + eps)
                - dy1 / (r8 * arg38 + eps)
            )

            byy /= 4 * np.pi

        if comp == "byy":

            rows += [byy]

        if comp == "bxx":

            rows += [bxx]

        if comp == "bzz":

            bzz = -bxx - byy
            rows += [bzz]

        if comp == "bxy":
            bxy = np.zeros((1, 3 * nC))

            bxy[0, 0:nC] = 2 * (
                ((dx1 * arg4) / (r1 * arg1 ** 2 + (dx1 ** 2) * r1 + eps))
                - ((dx2 * arg9) / (r2 * arg6 ** 2 + (dx2 ** 2) * r2 + eps))
                + ((dx2 * arg14) / (r3 * arg11 ** 2 + (dx2 ** 2) * r3 + eps))
                - ((dx1 * arg19) / (r4 * arg16 ** 2 + (dx1 ** 2) * r4 + eps))
                + ((dx2 * arg24) / (r5 * arg21 ** 2 + (dx2 ** 2) * r5 + eps))
                - ((dx1 * arg29) / (r6 * arg26 ** 2 + (dx1 ** 2) * r6 + eps))
                + ((dx1 * arg34) / (r7 * arg31 ** 2 + (dx1 ** 2) * r7 + eps))
                - ((dx2 * arg39) / (r8 * arg36 ** 2 + (dx2 ** 2) * r8 + eps))
            )
            bxy[0, nC : 2 * nC] = (
                dy2 / (r1 * arg5 + eps)
                - dy2 / (r2 * arg10 + eps)
                + dy2 / (r3 * arg15 + eps)
                - dy2 / (r4 * arg20 + eps)
                + dy1 / (r5 * arg25 + eps)
                - dy1 / (r6 * arg30 + eps)
                + dy1 / (r7 * arg35 + eps)
                - dy1 / (r8 * arg40 + eps)
            )
            bxy[0, 2 * nC :] = (
                1 / r1 - 1 / r2 + 1 / r3 - 1 / r4 + 1 / r5 - 1 / r6 + 1 / r7 - 1 / r8
            )

            bxy /= 4 * np.pi

            rows += [bxy]

        if comp == "bxz":
            bxz = np.zeros((1, 3 * nC))

            bxz[0, 0:nC] = 2 * (
                ((dx1 * arg5) / (r1 * (arg1 ** 2) + (dx1 ** 2) * r1 + eps))
                - ((dx2 * arg10) / (r2 * (arg6 ** 2) + (dx2 ** 2) * r2 + eps))
                + ((dx2 * arg15) / (r3 * (arg11 ** 2) + (dx2 ** 2) * r3 + eps))
                - ((dx1 * arg20) / (r4 * (arg16 ** 2) + (dx1 ** 2) * r4 + eps))
                + ((dx2 * arg25) / (r5 * (arg21 ** 2) + (dx2 ** 2) * r5 + eps))
                - ((dx1 * arg30) / (r6 * (arg26 ** 2) + (dx1 ** 2) * r6 + eps))
                + ((dx1 * arg35) / (r7 * (arg31 ** 2) + (dx1 ** 2) * r7 + eps))
                - ((dx2 * arg40) / (r8 * (arg36 ** 2) + (dx2 ** 2) * r8 + eps))
            )
            bxz[0, nC : 2 * nC] = (
                1 / r1 - 1 / r2 + 1 / r3 - 1 / r4 + 1 / r5 - 1 / r6 + 1 / r7 - 1 / r8
            )
            bxz[0, 2 * nC :] = (
                dz2 / (r1 * arg4 + eps)
                - dz2 / (r2 * arg9 + eps)
                + dz1 / (r3 * arg14 + eps)
                - dz1 / (r4 * arg19 + eps)
                + dz2 / (r5 * arg24 + eps)
                - dz2 / (r6 * arg29 + eps)
                + dz1 / (r7 * arg34 + eps)
                - dz1 / (r8 * arg39 + eps)
            )

            bxz /= 4 * np.pi

            rows += [bxz]

        if comp == "byz":
            byz = np.zeros((1, 3 * nC))

            byz[0, 0:nC] = (
                1 / r3 - 1 / r2 + 1 / r5 - 1 / r8 + 1 / r1 - 1 / r4 + 1 / r7 - 1 / r6
            )
            byz[0, nC : 2 * nC] = 2 * (
                ((dy2 * arg5) / (r1 * (arg2 ** 2) + (dy2 ** 2) * r1 + eps))
                - ((dy2 * arg10) / (r2 * (arg7 ** 2) + (dy2 ** 2) * r2 + eps))
                + ((dy2 * arg15) / (r3 * (arg12 ** 2) + (dy2 ** 2) * r3 + eps))
                - ((dy2 * arg20) / (r4 * (arg17 ** 2) + (dy2 ** 2) * r4 + eps))
                + ((dy1 * arg25) / (r5 * (arg22 ** 2) + (dy1 ** 2) * r5 + eps))
                - ((dy1 * arg30) / (r6 * (arg27 ** 2) + (dy1 ** 2) * r6 + eps))
                + ((dy1 * arg35) / (r7 * (arg32 ** 2) + (dy1 ** 2) * r7 + eps))
                - ((dy1 * arg40) / (r8 * (arg37 ** 2) + (dy1 ** 2) * r8 + eps))
            )
            byz[0, 2 * nC :] = (
                dz2 / (r1 * arg3 + eps)
                - dz2 / (r2 * arg8 + eps)
                + dz1 / (r3 * arg13 + eps)
                - dz1 / (r4 * arg18 + eps)
                + dz2 / (r5 * arg23 + eps)
                - dz2 / (r6 * arg28 + eps)
                + dz1 / (r7 * arg33 + eps)
                - dz1 / (r8 * arg38 + eps)
            )

            byz /= 4 * np.pi

            rows += [byz]

        if (comp == "bx") or ("tmi" in components):
            bx = np.zeros((1, 3 * nC))

            bx[0, 0:nC] = (
                (-2 * np.arctan2(dx1, arg1 + eps))
                - (-2 * np.arctan2(dx2, arg6 + eps))
                + (-2 * np.arctan2(dx2, arg11 + eps))
                - (-2 * np.arctan2(dx1, arg16 + eps))
                + (-2 * np.arctan2(dx2, arg21 + eps))
                - (-2 * np.arctan2(dx1, arg26 + eps))
                + (-2 * np.arctan2(dx1, arg31 + eps))
                - (-2 * np.arctan2(dx2, arg36 + eps))
            )
            bx[0, nC : 2 * nC] = (
                np.log(arg5)
                - np.log(arg10)
                + np.log(arg15)
                - np.log(arg20)
                + np.log(arg25)
                - np.log(arg30)
                + np.log(arg35)
                - np.log(arg40)
            )
            bx[0, 2 * nC :] = (
                (np.log(arg4) - np.log(arg9))
                + (np.log(arg14) - np.log(arg19))
                + (np.log(arg24) - np.log(arg29))
                + (np.log(arg34) - np.log(arg39))
            )
            bx /= -4 * np.pi

            # rows += [bx]

        if (comp == "by") or ("tmi" in components):
            by = np.zeros((1, 3 * nC))

            by[0, 0:nC] = (
                np.log(arg5)
                - np.log(arg10)
                + np.log(arg15)
                - np.log(arg20)
                + np.log(arg25)
                - np.log(arg30)
                + np.log(arg35)
                - np.log(arg40)
            )
            by[0, nC : 2 * nC] = (
                (-2 * np.arctan2(dy2, arg2 + eps))
                - (-2 * np.arctan2(dy2, arg7 + eps))
                + (-2 * np.arctan2(dy2, arg12 + eps))
                - (-2 * np.arctan2(dy2, arg17 + eps))
                + (-2 * np.arctan2(dy1, arg22 + eps))
                - (-2 * np.arctan2(dy1, arg27 + eps))
                + (-2 * np.arctan2(dy1, arg32 + eps))
                - (-2 * np.arctan2(dy1, arg37 + eps))
            )
            by[0, 2 * nC :] = (
                (np.log(arg3) - np.log(arg8))
                + (np.log(arg13) - np.log(arg18))
                + (np.log(arg23) - np.log(arg28))
                + (np.log(arg33) - np.log(arg38))
            )

            by /= -4 * np.pi

            # rows += [by]

        if (comp == "bz") or ("tmi" in components):
            bz = np.zeros((1, 3 * nC))

            bz[0, 0:nC] = (
                np.log(arg4)
                - np.log(arg9)
                + np.log(arg14)
                - np.log(arg19)
                + np.log(arg24)
                - np.log(arg29)
                + np.log(arg34)
                - np.log(arg39)
            )
            bz[0, nC : 2 * nC] = (
                (np.log(arg3) - np.log(arg8))
                + (np.log(arg13) - np.log(arg18))
                + (np.log(arg23) - np.log(arg28))
                + (np.log(arg33) - np.log(arg38))
            )
            bz[0, 2 * nC :] = (
                (-2 * np.arctan2(dz2, arg1_ + eps))
                - (-2 * np.arctan2(dz2, arg6_ + eps))
                + (-2 * np.arctan2(dz1, arg11_ + eps))
                - (-2 * np.arctan2(dz1, arg16_ + eps))
                + (-2 * np.arctan2(dz2, arg21_ + eps))
                - (-2 * np.arctan2(dz2, arg26_ + eps))
                + (-2 * np.arctan2(dz1, arg31_ + eps))
                - (-2 * np.arctan2(dz1, arg36_ + eps))
            )
            bz /= -4 * np.pi

        if comp == "bx":

            rows += [bx]

        if comp == "by":

            rows += [by]

        if comp == "bz":

            rows += [bz]

        if comp == "tmi":

            rows += [np.dot(P, np.r_[bx, by, bz])]

    return np.vstack(rows)


def progress(iter, prog, final):
    """
    progress(iter,prog,final)

    Function measuring the progress of a process and print to screen the %.
    Useful to estimate the remaining runtime of a large problem.

    Created on Dec, 20th 2015

    @author: dominiquef
    """
    arg = np.floor(float(iter) / float(final) * 10.0)

    if arg > prog:

        print("Done " + str(arg * 10) + " %")
        prog = arg

    return prog


def get_dist_wgt(mesh, rxLoc, actv, R, R0):
    """
    get_dist_wgt(xn,yn,zn,rxLoc,R,R0)

    Function creating a distance weighting function required for the magnetic
    inverse problem.

    INPUT
    xn, yn, zn : Node location
    rxLoc       : Observation locations [obsx, obsy, obsz]
    actv        : Active cell vector [0:air , 1: ground]
    R           : Decay factor (mag=3, grav =2)
    R0          : Small factor added (default=dx/4)

    OUTPUT
    wr       : [nC] Vector of distance weighting

    Created on Dec, 20th 2015

    @author: dominiquef
    """

    # Find non-zero cells
    if actv.dtype == "bool":
        inds = (
            np.asarray([inds for inds, elem in enumerate(actv, 1) if elem], dtype=int)
            - 1
        )
    else:
        inds = actv

    nC = len(inds)

    # Create active cell projector
    P = csr((np.ones(nC), (inds, range(nC))), shape=(mesh.nC, nC))

    # Geometrical constant
    p = 1 / np.sqrt(3)

    # Create cell center location
    Ym, Xm, Zm = np.meshgrid(mesh.vectorCCy, mesh.vectorCCx, mesh.vectorCCz)
    hY, hX, hZ = np.meshgrid(mesh.hy, mesh.hx, mesh.hz)

    # Remove air cells
    Xm = P.T * mkvc(Xm)
    Ym = P.T * mkvc(Ym)
    Zm = P.T * mkvc(Zm)

    hX = P.T * mkvc(hX)
    hY = P.T * mkvc(hY)
    hZ = P.T * mkvc(hZ)

    V = P.T * mkvc(mesh.vol)
    wr = np.zeros(nC)

    ndata = rxLoc.shape[0]
    count = -1
    print("Begin calculation of distance weighting for R= " + str(R))

    for dd in range(ndata):

        nx1 = (Xm - hX * p - rxLoc[dd, 0]) ** 2
        nx2 = (Xm + hX * p - rxLoc[dd, 0]) ** 2

        ny1 = (Ym - hY * p - rxLoc[dd, 1]) ** 2
        ny2 = (Ym + hY * p - rxLoc[dd, 1]) ** 2

        nz1 = (Zm - hZ * p - rxLoc[dd, 2]) ** 2
        nz2 = (Zm + hZ * p - rxLoc[dd, 2]) ** 2

        R1 = np.sqrt(nx1 + ny1 + nz1)
        R2 = np.sqrt(nx1 + ny1 + nz2)
        R3 = np.sqrt(nx2 + ny1 + nz1)
        R4 = np.sqrt(nx2 + ny1 + nz2)
        R5 = np.sqrt(nx1 + ny2 + nz1)
        R6 = np.sqrt(nx1 + ny2 + nz2)
        R7 = np.sqrt(nx2 + ny2 + nz1)
        R8 = np.sqrt(nx2 + ny2 + nz2)

        temp = (
            (R1 + R0) ** -R
            + (R2 + R0) ** -R
            + (R3 + R0) ** -R
            + (R4 + R0) ** -R
            + (R5 + R0) ** -R
            + (R6 + R0) ** -R
            + (R7 + R0) ** -R
            + (R8 + R0) ** -R
        )

        wr = wr + (V * temp / 8.0) ** 2.0

        count = progress(dd, count, ndata)

    wr = np.sqrt(wr) / V
    wr = mkvc(wr)
    wr = np.sqrt(wr / (np.max(wr)))

    print("Done 100% ...distance weighting completed!!\n")

    return wr
