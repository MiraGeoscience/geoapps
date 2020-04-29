from __future__ import print_function
from .. import Problem, Mesh
from .. import Utils
from ..Utils import mkvc, matutils, sdiag
from .. import Props
import scipy as sp
import scipy.constants as constants
import os
import numpy as np
import dask
import dask.array as da
from scipy.sparse import csr_matrix as csr
from dask.diagnostics import ProgressBar
import multiprocessing

class GravityIntegral(Problem.LinearProblem):

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    forwardOnly = False  # If false, matrix is store to memory (watch your RAM)
    actInd = None  #: Active cell indices provided
    verbose = True
    gtgdiag = None
    n_cpu = None
    parallelized = True
    progressIndex = -1
    max_chunk_size = None
    chunk_by_rows = False
    Jpath = "./sensitivity.zarr"
    maxRAM = 8  # Maximum memory usage


    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        if getattr(self, 'actInd', None) is not None:

            if self.actInd.dtype == 'bool':
                inds = np.asarray([inds for inds,
                                  elem in enumerate(self.actInd, 1) if elem],
                                  dtype=int) - 1
            else:
                inds = self.actInd

        else:

            inds = np.asarray(range(self.mesh.nC))

        self.nC = len(inds)

        # Create active cell projector
        P = csr((np.ones(self.nC), (inds, range(self.nC))),
                          shape=(self.mesh.nC, self.nC))

        # Create vectors of nodal location
        # (lower and upper corners for each cell)
        bsw = (self.mesh.gridCC - self.mesh.h_gridded/2.)
        tne = (self.mesh.gridCC + self.mesh.h_gridded/2.)

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]

        self.Yn = P.T*np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = P.T*np.c_[mkvc(xn1), mkvc(xn2)]

        if self.mesh.dim > 2:
            zn1, zn2 = bsw[:, 2], tne[:, 2]
            self.Zn = P.T*np.c_[mkvc(zn1), mkvc(zn2)]

    def fields(self, m):
        # self.model = self.rhoMap*m

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense G
            return np.array(self.Intrgl_Fwr_Op(m=m), dtype='float')

        else:
            # fields = da.dot(self.G, m)

            return da.dot(self.G, (self.rhoMap*m).astype(np.float32)) #np.array(fields, dtype='float')

    def modelMap(self):
        """
            Call for general mapping of the problem
        """
        return self.rhoMap

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """

        dmudm = self.rhoMap.deriv(m)
        self.model = m

        if self.gtgdiag is None:

            if W is None:
                W = np.ones(self.G.shape[1])

            self.gtgdiag = da.sum(da.power(W[:, None].astype(np.float32) * self.G, 2), axis=0).compute()

            # for ii in range(self.G.shape[0]):

            #     self.gtgdiag += (w[ii]*self.G[ii, :]*dmudm)**2.

        return mkvc(np.sum((sdiag(mkvc(self.gtgdiag)**0.5) * dmudm).power(2.), axis=0))

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """

        dmudm = self.rhoMap.deriv(m)

        return da.dot(self.G, dmudm)

    def Jvec(self, m, v, f=None):
        dmudm = self.rhoMap.deriv(m)

        dmudm_v = da.from_array(dmudm*v, chunks=self.G.chunks[1])

        return da.dot(self.G, dmudm_v.astype(np.float32))

    def Jtvec(self, m, v, f=None):

        dmudm = self.rhoMap.deriv(m)
        Jtvec = da.dot(v.astype(np.float32), self.G)
        dmudm_v = dask.delayed(csr.dot)(Jtvec, dmudm)

        return da.from_delayed(dmudm_v, dtype=float, shape=[dmudm.shape[1]])

    @property
    def G(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_G', None) is None:

            self._G = self.Intrgl_Fwr_Op()

        return self._G

    def Intrgl_Fwr_Op(self, m=None):

        """

        Gravity forward operator in integral form

        flag        = 'z' | 'xyz'

        Return
        _G        = Linear forward modeling operation

        Created on March, 15th 2016

        @author: dominiquef

         """
        if m is not None:
            self.model = self.rhoMap*m

        print("Multiplication")
        self.rxLoc = self.survey.srcField.rxList[0].locs
        self.nD = int(self.rxLoc.shape[0])

        # if self.n_cpu is None:
        #     self.n_cpu = multiprocessing.cpu_count()

        # Switch to determine if the process has to be run in parallel
        job = Forward(
                rxLoc=self.rxLoc, Xn=self.Xn, Yn=self.Yn, Zn=self.Zn,
                n_cpu=self.n_cpu, forwardOnly=self.forwardOnly,
                model=self.model, components=self.survey.components,
                parallelized=self.parallelized,
                verbose=self.verbose, Jpath=self.Jpath, maxRAM=self.maxRAM,
                max_chunk_size=self.max_chunk_size, chunk_by_rows=self.chunk_by_rows
                )

        G = job.calculate()

        return G


class Forward(object):
    """
        Add docstring once it works
    """

    progressIndex = -1
    parallelized = True
    rxLoc = None
    Xn, Yn, Zn = None, None, None
    n_cpu = None
    forwardOnly = False
    model = None
    components = ['gz']

    verbose = True
    maxRAM = 1
    chunk_by_rows = False

    max_chunk_size = None
    Jpath = "./sensitivity.zarr"

    def __init__(self, **kwargs):
        super(Forward, self).__init__()
        Utils.setKwargs(self, **kwargs)

    def calculate(self):

        self.nD = self.rxLoc.shape[0]
        self.nC = self.Xn.shape[0]

        if self.n_cpu is None:
            self.n_cpu = int(multiprocessing.cpu_count())

        # Set this early so we can get a better memory estimate for dask chunking
        nDataComps = len(self.components)

        if self.parallelized:

            row = dask.delayed(self.calcTrow, pure=True)

            makeRows = [row(self.rxLoc[ii, :]) for ii in range(self.nD)]

            buildMat = [da.from_delayed(makeRow, dtype=np.float32, shape=(nDataComps,  self.nC)) for makeRow in makeRows]

            stack = da.vstack(buildMat)

            # Auto rechunk
            # To customise memory use set Dask config in calling scripts: dask.config.set({'array.chunk-size': '128MiB'})
            if self.forwardOnly or self.chunk_by_rows:
                label = 'DASK: Chunking by rows'
                # Autochunking by rows is faster and more memory efficient for
                # very large problems sensitivty and forward calculations
                target_size = dask.config.get('array.chunk-size').replace('MiB',' MB')
                stack = stack.rechunk({0: 'auto', 1: -1})
            elif self.max_chunk_size:
                label = 'DASK: Chunking using parameters'
                # Manual chunking is less sensitive to chunk sizes for some problems
                target_size = "{:.0f} MB".format(self.max_chunk_size)
                nChunks_col = 1
                nChunks_row = 1
                rowChunk = int(np.ceil(stack.shape[0]/nChunks_row))
                colChunk = int(np.ceil(stack.shape[1]/nChunks_col))
                chunk_size = rowChunk*colChunk*8*1e-6  # in Mb

                # Add more chunks until memory falls below target
                while chunk_size >= self.max_chunk_size:

                    if rowChunk > colChunk:
                        nChunks_row += 1
                    else:
                        nChunks_col += 1

                    rowChunk = int(np.ceil(stack.shape[0]/nChunks_row))
                    colChunk = int(np.ceil(stack.shape[1]/nChunks_col))
                    chunk_size = rowChunk*colChunk*8*1e-6  # in Mb

                stack = stack.rechunk((rowChunk, colChunk))
            else:
                label = 'DASK: Chunking by columns'
                # Autochunking by columns is faster for Inversions
                target_size = dask.config.get('array.chunk-size').replace('MiB',' MB')
                stack = stack.rechunk({0: -1, 1: 'auto'})

            if self.verbose:
                print(label)
                print('Tile size (nD, nC): ', stack.shape)
    #                print('Chunk sizes (nD, nC): ', stack.chunks) # For debugging only
                print('Number of chunks: %.0f x %.0f = %.0f' %
                    (len(stack.chunks[0]), len(stack.chunks[1]), len(stack.chunks[0]) * len(stack.chunks[1])))
                print("Target chunk size: %s" % target_size)
                print('Max chunk size %.0f x %.0f = %.3f MB' % (max(stack.chunks[0]), max(stack.chunks[1]), max(stack.chunks[0]) * max(stack.chunks[1]) * 8*1e-6))
                print('Min chunk size %.0f x %.0f = %.3f MB' % (min(stack.chunks[0]), min(stack.chunks[1]), min(stack.chunks[0]) * min(stack.chunks[1]) * 8*1e-6))
                print('Max RAM (GB x %.0f CPU): %.6f' %
                    (self.n_cpu, max(stack.chunks[0]) * max(stack.chunks[1]) * 8*1e-9 * self.n_cpu))
                print('Tile size (GB): %.3f' % (stack.shape[0] * stack.shape[1] * 8*1e-9))

            if self.forwardOnly:

                with ProgressBar():
                    print("Forward calculation: ")
                    pred = da.dot(stack, self.model).compute()

                return pred

            else:

                if os.path.exists(self.Jpath):

                    G = da.from_zarr(self.Jpath)

                    if np.all(np.r_[
                            np.any(np.r_[G.chunks[0]] == stack.chunks[0]),
                            np.any(np.r_[G.chunks[1]] == stack.chunks[1]),
                            np.r_[G.shape] == np.r_[stack.shape]]):
                        # Check that loaded G matches supplied data and mesh
                        print("Zarr file detected with same shape and chunksize ... re-loading")

                        return G
                    else:

                        print("Zarr file detected with wrong shape and chunksize ... over-writing")

                with ProgressBar():
                    print("Saving G to zarr: " + self.Jpath)
                    G = da.to_zarr(stack, self.Jpath, compute=True, return_stored=True, overwrite=True)

        else:

            result = []
            for ii in range(self.nD):
                result += [self.calcTrow(self.rxLoc[ii, :])]
                self.progress(ii, self.nD)

            G = np.vstack(result)

        return G

    def calcTrow(self, receiver_location):
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
        eps = 1e-8

        NewtG = constants.G*1e+8

        dx = self.Xn - receiver_location[0]
        dy = self.Yn - receiver_location[1]
        dz = self.Zn - receiver_location[2]

        compDict = {key: np.zeros(self.Xn.shape[0]) for key in self.components}

        gxx = np.zeros(self.Xn.shape[0])
        gyy = np.zeros(self.Xn.shape[0])

        for aa in range(2):
            for bb in range(2):
                for cc in range(2):

                    r = (
                            mkvc(dx[:, aa]) ** 2 +
                            mkvc(dy[:, bb]) ** 2 +
                            mkvc(dz[:, cc]) ** 2
                        ) ** (0.50) + eps

                    dz_r = dz[:, cc] + r + eps
                    dy_r = dy[:, bb] + r + eps
                    dx_r = dx[:, aa] + r + eps

                    dxr = dx[:, aa] * r + eps
                    dyr = dy[:, bb] * r + eps
                    dzr = dz[:, cc] * r + eps

                    dydz = dy[:, bb] * dz[:, cc]
                    dxdy = dx[:, aa] * dy[:, bb]
                    dxdz = dx[:, aa] * dz[:, cc]

                    if 'gx' in self.components:
                        compDict['gx'] += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dy[:, bb] * np.log(dz_r) +
                            dz[:, cc] * np.log(dy_r) -
                            dx[:, aa] * np.arctan(dydz /
                                                  dxr)
                        )

                    if 'gy' in self.components:
                        compDict['gy']  += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dz_r) +
                            dz[:, cc] * np.log(dx_r) -
                            dy[:, bb] * np.arctan(dxdz /
                                                  dyr)
                        )

                    if 'gz' in self.components:
                        compDict['gz']  += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dy_r) +
                            dy[:, bb] * np.log(dx_r) -
                            dz[:, cc] * np.arctan(dxdy /
                                                  dzr)
                        )

                    arg = dy[:, bb] * dz[:, cc] / dxr

                    if ('gxx' in self.components) or ("gzz" in self.components) or ("guv" in self.components):
                        gxx -= 1e+4 * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r * dz_r + eps) +
                            dxdz / (r * dy_r + eps) -
                            np.arctan(arg+eps) +
                            dx[:, aa] * (1./ (1+arg**2.)) *
                            dydz/dxr**2. *
                            (r + dx[:, aa]**2./r)
                        )

                    if 'gxy' in self.components:
                        compDict['gxy'] -= 1e+4 * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dz_r) + dy[:, bb]**2./ (r*dz_r) +
                            dz[:, cc] / r  -
                            1. / (1+arg**2.+ eps) * (dz[:, cc]/r**2) * (r - dy[:, bb]**2./r)

                        )

                    if 'gxz' in self.components:
                        compDict['gxz'] -= 1e+4 * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dy_r) + dz[:, cc]**2./ (r*dy_r) +
                            dy[:, bb] / r  -
                            1. / (1+arg**2.) * (dy[:, bb]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

                    arg = dx[:, aa]*dz[:, cc]/dyr

                    if ('gyy' in self.components) or ("gzz" in self.components) or ("guv" in self.components):
                        gyy -= 1e+4 * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r*dz_r+ eps) +
                            dydz / (r*dx_r+ eps) -
                            np.arctan(arg+eps) +
                            dy[:, bb] * (1./ (1+arg**2.+ eps)) *
                            dxdz/dyr**2. *
                            (r + dy[:, bb]**2./r)
                        )

                    if 'gyz' in self.components:
                        compDict['gyz'] -= 1e+4 * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dx_r) + dz[:, cc]**2./ (r*(dx_r)) +
                            dx[:, aa] / r  -
                            1. / (1+arg**2.) * (dx[:, aa]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

        if 'gyy' in self.components:
            compDict['gyy'] = gyy

        if 'gxx' in self.components:
            compDict['gxx'] = gxx

        if 'gzz' in self.components:
            compDict['gzz'] = -gxx - gyy

        if 'guv' in self.components:
            compDict['guv'] = -0.5*(gxx - gyy)

        return np.vstack([NewtG * compDict[key] for key in list(compDict.keys())])


class Problem3D_Diff(Problem.BaseProblem):
    """
        Gravity in differential equations!
    """

    _depreciate_main_map = 'rhoMap'

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    solver = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        self.mesh.setCellGradBC('dirichlet')

        self._Div = self.mesh.cellGrad

    @property
    def MfI(self): return self._MfI

    @property
    def Mfi(self): return self._Mfi

    def makeMassMatrices(self, m):
        self.model = m
        self._Mfi = self.mesh.getFaceInnerProduct()
        self._MfI = Utils.sdiag(1. / self._Mfi.diagonal())

    def getRHS(self, m):
        """


        """

        Mc = Utils.sdiag(self.mesh.vol)

        self.model = m
        rho = self.rho

        return Mc * rho

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Gravity nodal problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return -self._Div.T * self.Mfi * self._Div

    def fields(self, m):
        """
            Return magnetic potential (u) and flux (B)
            u: defined on the cell nodes [nC x 1]
            gField: defined on the cell faces [nF x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        from scipy.constants import G as NewtG

        self.makeMassMatrices(m)
        A = self.getA(m)
        RHS = self.getRHS(m)

        if self.solver is None:
            m1 = sp.linalg.interface.aslinearoperator(
                Utils.sdiag(1 / A.diagonal())
            )
            u, info = sp.linalg.bicgstab(A, RHS, tol=1e-6, maxiter=1000, M=m1)

        else:
            print("Solving with Paradiso")
            Ainv = self.solver(A)
            u = Ainv * RHS

        gField = 4. * np.pi * NewtG * 1e+8 * self._Div * u

        return {'G': gField, 'u': u}
