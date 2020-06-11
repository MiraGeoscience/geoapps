module rTE_Fortran

implicit none

complex(kind=8), parameter :: j = complex(0.d0, 1.d0)
real(kind=8), parameter :: pi = dacos(-1.d0)
real(kind=8), parameter :: pi2 = 2.d0*pi
real(kind=8), parameter :: mu0 = 4.d-7*pi ! H/m

private

public :: rTE_forward
public :: rTE_sensitivity

contains

!====================================================================!
subroutine rTE_forward(nLayers, nFrequencies, nFilter, frequencies, lambda, sig, chi, depth, halfSpace, rTE)
  !! Computes the TE portion for forward modelling. Callable from python.
!====================================================================!
  integer, intent(in) :: nLayers
  integer, intent(in) :: nFrequencies
  integer, intent(in) :: nFilter
  real(kind=8), intent(in) :: frequencies(nFrequencies, nFilter)
  !f2py intent(in) :: frequencies
  real(kind=8), intent(in) :: lambda(nFrequencies, nFilter)
  !f2py intent(in) :: lambda
  complex(kind=8), intent(in) :: sig(nLayers, nFrequencies, nFilter)
  !f2py intent(in) :: sig
  real(kind=8), intent(in) :: chi(nLayers)
  !f2py intent(in) :: chi
  real(kind=8), intent(in) :: depth(nLayers)
  !f2py intent(in) :: depth
  logical, intent(in) :: halfspace
  !f2py intent(in) :: halfSpace
  complex(kind=8), intent(inout) :: rTE(nFrequencies, nFilter)
  !f2py intent(in, out) :: rTE

  integer :: i, jj, k
  complex(kind=8) :: c, cm1, cp1
  complex(kind=8) :: m0, m1, m2, m3
  complex(kind=8) :: s0, s1, s2, s3
  complex(kind=8) :: ss0, ss1, ss2, ss3
  real(kind=8) :: h
  real(kind=8) :: lam2
  real(kind=8) :: omega
  real(kind=8) :: tmp0
  complex(kind=8) :: uTmp0, uTmp1, cTmp
  real(kind=8) :: thickness(nLayers - 1)
  complex(kind=8) :: c1(nLayers)

  if (halfSpace .or. nLayers == 1) then

    cTmp = mu0 * (1.d0 + chi(1))
    do jj = 1, nFilter
      do i = 1, nFrequencies
        omega = pi2 * frequencies(i, jj)
        uTmp0 = lambda(i, jj)
        uTmp1 = sqrt(uTmp0**2.d0 + j * omega * cTmp * sig(1, i, jj))
        c = mu0 * uTmp1 / (cTmp * uTmp0)

        cm1 = 0.5d0 * (1.d0 - c)
        cp1 = 0.5d0 * (1.d0 + c)

        rTE(i, jj) = cm1 / cp1
      enddo
    enddo

    return ! Early escape
  endif


  do k = 1, nLayers - 1
    thickness(k) = -(depth(k+1) - depth(k))
    c1(k) = 1.d0 + chi(k)
  enddo
  c1(nLayers) = 1.d0 + chi(nLayers)

  do jj = 1, nFilter
    do i = 1, nFrequencies
      omega = pi2 * frequencies(i, jj)
      tmp0 = omega * mu0
      lam2 = lambda(i, jj)**2.d0

      ! Set up first layer
      uTmp1 = sqrt(lam2 + j * tmp0 * c1(1) * sig(1, i, jj))
      c = uTmp1 / (c1(1) * lambda(i, jj))

      s0 = 0.5d0 * (1.d0 + c)
      s1 = 0.5d0 * (1.d0 - c)
      s2 = s1
      s3 = s0

      do k = 1, nLayers - 1
        cTmp = j * tmp0
        uTmp0 = sqrt(lam2 + cTmp * c1(k) * sig(k, i, jj))
        uTmp1 = sqrt(lam2 + cTmp * c1(k + 1) * sig(k+1, i, jj))
        c = (c1(k) * uTmp1) / (c1(k + 1) * uTmp0)

        h = thickness(k)

        cTmp = exp(-2.d0 * uTmp0 * h)
        cm1 = 0.5d0 * (1.d0 - c)
        cp1 = 0.5d0 * (1.d0 + c)

        m0 = cp1 * cTmp
        m1 = cm1
        m2 = cm1 * cTmp
        m3 = cp1

        ss0 = (s0 * m0) + (s2 * m1)
        ss1 = (s1 * m0) + (s3 * m1)
        ss2 = (s0 * m2) + (s2 * m3)
        ss3 = (s1 * m2) + (s3 * m3)

        s0 = ss0
        s1 = ss1
        s2 = ss2
        s3 = ss3
      enddo

      rTE(i, jj) = s2 / s3
    enddo
  enddo

  end subroutine
  !====================================================================!

  !====================================================================!
  subroutine rTE_sensitivity(nLayers, nFrequencies, nFilter, frequencies, lambda, sig, chi, depth, halfSpace, drTE)
  !! Computes the TE portion for forward modelling. Callable from python.
  !====================================================================!
  integer, intent(in) :: nLayers
  integer, intent(in) :: nFrequencies
  integer, intent(in) :: nFilter
  real(kind=8), intent(in) :: frequencies(nFrequencies, nFilter)
  !f2py intent(in) :: frequencies
  real(kind=8), intent(in) :: lambda(nFrequencies, nFilter)
  !f2py intent(in) :: lambda
  complex(kind=8), intent(in) :: sig(nLayers, nFrequencies, nFilter)
  !f2py intent(in) :: sig
  real(kind=8), intent(in) :: chi(nLayers)
  !f2py intent(in) :: chi
  real(kind=8), intent(in) :: depth(nLayers)
  !f2py intent(in) :: depth
  logical, intent(in) :: halfspace
  !f2py intent(in) :: halfSpace
  complex(kind=8), intent(inout) :: drTE(nLayers, nFrequencies, nFilter)
  !f2py intent(in, out) :: drTE

  real(kind=8) :: h0, h_1
  real(kind=8) :: lam, lam2
  real(kind=8) :: thickness(nLayers - 1)
  real(kind=8) :: c1(nLayers), c1t, c2t, c3t

  complex(kind=8) :: const, cTmp, cTmp2, cTmp3, cTmp4
  complex(kind=8) :: dJ_10mTemp00, dJ_10mTemp01, dJ_10mTemp10, dJ_10mTemp11
  real(kind=8)    :: dJ0mTemp00_
  complex(kind=8) :: dJ0mTemp00, dJ0mTemp11
  complex(kind=8) :: dJ01mTemp00, dJ01mTemp01, dJ01mTemp10, dJ01mTemp11
  complex(kind=8) :: dJ1mTemp00, dJ1mTemp01, dJ1mTemp10
  complex(kind=8) :: dJ0sum00, dJ0sum10, dJ0sum01, dJ0sum11
  complex(kind=8) :: dJ1sum00, dJ1sum10, dJ1sum01, dJ1sum11
  complex(kind=8) :: dudsig
  complex(kind=8) :: mTemp00, mTemp01, mTemp10, mTemp11
  complex(kind=8) :: m0Sum00, m0Sum01, m0Sum10, m0Sum11
  complex(kind=8) :: m1Sum00, m1Sum01, m1Sum10, m1Sum11
  complex(kind=8) :: w
  complex(kind=8) :: utemp0, utemp1
  complex(kind=8), dimension(nLayers) :: M00, M01, M10, M11
  complex(kind=8), dimension(nLayers) :: dJ00, dJ01, dJ10, dJ11

  integer :: i
  integer :: jj
  integer :: k, k1

  ! Half space model, provide early exit
  if (halfspace .or. nLayers == 1) then
    cTmp = 1.d0 + chi(1)
    do jj = 1, nFilter
      do i = 1, nFrequencies

        w = pi2 * frequencies(i, jj)
        lam = lambda(i, jj)
        cTmp2 = j * w * mu0 * cTmp
        cTmp3 = 1.d0 / (cTmp * lam)

        ! utemp0 = lambda
        utemp1 = sqrt(lam**2.d0 + cTmp2 * sig(1, i, jj))
        const = utemp1 * cTmp3

        ! Compute M1
        M0sum01 = 0.5d0 * (1.d0 - const)
        M0sum11 = 0.5d0 * (1.d0 + const)

        ! Compute dM1du1
        dJ1sum11 = 0.25d0 * (cTmp2 / utemp1) * cTmp3

        drTE(1, i, jj) = ((-dJ1sum11 * M0sum11) - dJ1sum11 * M0sum01) / M0sum11**2.d0

      enddo
    enddo
    return ! Early Exit
  endif

  do k = 1, nLayers - 1
    thickness(k) = -(depth(k+1) - depth(k))
    c1(k) = 1.d0 + chi(k)
  enddo
  c1(nLayers) = 1.d0 + chi(nLayers)

  do jj = 1, nFilter
    do i = 1, nFrequencies

      c1t = c1(1)
      c2t = c1(2)

      lam = lambda(i, jj)
      lam2 = lam**2.d0
      w = pi2 * frequencies(i, jj)
      c3t = 1.d0 / (c1t * lam)
      cTmp2 = j * w * mu0

      ! utemp0 = lambda
      utemp1 = sqrt(lam2 + cTmp2 * c1t * sig(1, i, jj))

      const = utemp1 * c3t

      ! Compute M1
      Mtemp00 = 0.5d0 * (1.d0 + const)
      Mtemp10 = 0.5d0 * (1.d0 - const)
      ! Mtemp01 = mTemp10
      ! Mtemp11 = mTemp00

      M00(1) = Mtemp00
      M01(1) = Mtemp10
      M10(1) = Mtemp10
      M11(1) = Mtemp00

      M0sum00 = Mtemp00
      M0sum10 = Mtemp10

      ! Compute dM1du1
      dj0Mtemp00_ =  0.5d0 * c3t

      !!!!!!!!
      ! j = 0
      !!!!!!!!

      utemp0  = utemp1
      utemp1  = sqrt(lam2 + cTmp2 * c2t * sig(2, i, jj))
      const =  (c1t * utemp1) / (c2t * utemp0)

      h0 = thickness(1)

      mTemp11 = 0.5d0 * (1.d0 + const)
      mTemp10 = 0.5d0 * (1.d0 - const)
      mTemp00 = exp(-2.d0 * utemp0 * h0)
      mTemp01 = mTemp10 * mTemp00
      mTemp00 = mTemp11 * mTemp00

      M00(2) = Mtemp00
      M01(2) = Mtemp01
      M10(2) = Mtemp10
      M11(2) = Mtemp11

      ! Inline the matmul
      M1sum10 = (M0sum10 * Mtemp00) + (M0sum00 * Mtemp10)
      M1sum00 = (M0sum00 * Mtemp00) + (M0sum10 * Mtemp10)
      M1sum01 = (M0sum00 * Mtemp01) + (M0sum10 * Mtemp11)
      M1sum11 = (M0sum10 * Mtemp01) + (M0sum00 * Mtemp11)

      M0sum00 = M1sum00
      M0sum10 = M1sum10
      M0sum01 = M1sum01
      M0sum11 = M1sum11

      ! TODO: for Computing Jacobian
      dudsig = 0.5d0 * cTmp2 * c1(1) / utemp0

      dJ01Mtemp00 = (c1t * utemp1) / (c2t * utemp0**2.d0)
      dJ01Mtemp11 = dJ01Mtemp00 * utemp0
      dJ01Mtemp01 = exp(-2.d0 * utemp0 * h0)

      ! Compute dM1dm1*M2
      dJ_10Mtemp00 = dj0Mtemp00_ * (Mtemp00 - Mtemp10)
      ! dJ_10Mtemp10 = -dJ_10Mtemp00
      dJ_10Mtemp01 = dj0Mtemp00_ * (Mtemp01 - Mtemp11)
      ! dJ_10Mtemp11 = -dJ_10Mtemp01

      dj1Mtemp10 =  0.5d0 * dJ01Mtemp00
      cTmp3 = h0 * dJ01Mtemp01
      dj1Mtemp01 =  dj1Mtemp10 * dJ01Mtemp01
      dj1Mtemp00 = -dj1Mtemp01 - (1.d0 + dJ01Mtemp11) * cTmp3
      dj1Mtemp01 =  dj1Mtemp01 - (1.d0 - dJ01Mtemp11) * cTmp3
      ! dj1Mtemp11 = -dj1Mtemp10

      ! Compute M1*dM2dm1  ! Re-coded given "symmetry"
      cTmp4 = M01(1) * dj1Mtemp10
      dJ01Mtemp00 = (M00(1) * dj1Mtemp00) + cTmp4
      dJ01Mtemp01 = (M00(1) * dj1Mtemp01) - cTmp4
      cTmp4 = M11(1) * dj1Mtemp10
      dJ01Mtemp10 = (M10(1) * dj1Mtemp00) + cTmp4
      dJ01Mtemp11 = (M10(1) * dj1Mtemp01) - cTmp4

      dJ00(1) = dudsig * (dJ_10Mtemp00 + dJ01Mtemp00)
      dJ10(1) = dudsig * (dJ01Mtemp10 - dJ_10Mtemp00)
      dJ01(1) = dudsig * (dJ_10Mtemp01 + dJ01Mtemp01)
      dJ11(1) = dudsig * (dJ01Mtemp11 - dJ_10Mtemp01)

      do k = 2, nLayers - 1
        c1t = c1(k - 1)
        c2t = c1(k)
        c3t = c1(k + 1)
        utemp0  = utemp1
        utemp1  = sqrt(lam2 + cTmp2 * c3t * sig(k + 1, i, jj))
        const =  (c2t * utemp1) / (c3t * utemp0)

        h0 = thickness(k)

        mTemp11 = 0.5d0 * (1.d0 + const)
        mTemp10 = 0.5d0 * (1.d0 - const)
        mTemp00 = exp(-2.d0 * utemp0 * h0)
        mTemp01 = mTemp10 * mTemp00
        mTemp00 = mTemp11 * mTemp00

        M00(k + 1) = Mtemp00
        M01(k + 1) = Mtemp01
        M10(k + 1) = Mtemp10
        M11(k + 1) = Mtemp11

        ! Inline the matmul
        M1sum00 = (M0sum00 * Mtemp00) + (M0sum01 * Mtemp10)
        M1sum10 = (M0sum10 * Mtemp00) + (M0sum11 * Mtemp10)
        M1sum01 = (M0sum00 * Mtemp01) + (M0sum01 * Mtemp11)
        M1sum11 = (M0sum10 * Mtemp01) + (M0sum11 * Mtemp11)

        M0sum00 = M1sum00
        M0sum10 = M1sum10
        M0sum01 = M1sum01
        M0sum11 = M1sum11

        ! TODO: for Computing Jacobian
        dudsig = 0.5d0 * cTmp2 * c2t / utemp0

        h_1 = thickness(k - 1)
        dJ_10Mtemp00 = sqrt(lam**2.0 + cTmp2 * c1t * sig(k - 1, i, jj))
        dJ_10Mtemp10 = c1t / (c2t * dJ_10Mtemp00)
        dJ_10Mtemp11 = exp(-2.d0 * dJ_10Mtemp00 * h_1)

        dj0Mtemp11 =  0.5 * dJ_10Mtemp10
        dj0Mtemp00 =  dj0Mtemp11 * dJ_10Mtemp11

        ! Compute dMjdmj*Mj+1  ! Re-coded given "symmetry"
        cTmp4 = Mtemp00 - Mtemp10
        dJ_10Mtemp00 = dj0Mtemp00 * cTmp4
        dJ_10Mtemp10 = -dj0Mtemp11 * cTmp4
        cTmp4 = Mtemp01 - Mtemp11
        dJ_10Mtemp01 = dj0Mtemp00 * cTmp4
        dJ_10Mtemp11 = -dj0Mtemp11 * cTmp4

        dJ01Mtemp00 = (c2t * utemp1) / (c3t * utemp0**2.d0)
        dJ01Mtemp11 = dJ01Mtemp00 * utemp0
        dJ01Mtemp10 = exp(-2.d0 * utemp0 * h0)

        dj1Mtemp10 =  0.5*dJ01Mtemp00
        cTmp3 = h0 * dJ01Mtemp10
        dj1Mtemp01 =  dj1Mtemp10 * dJ01Mtemp10
        dj1Mtemp00 = -dj1Mtemp01 - (1.d0 + dJ01Mtemp11) * cTmp3
        dj1Mtemp01 =  dj1Mtemp01 - (1.d0 - dJ01Mtemp11) * cTmp3
        ! dj1Mtemp11 = -dj1Mtemp10

        ! Compute Mj*dMj+1dmj
        cTmp4 = M01(k) * dj1Mtemp10
        dJ01Mtemp00 = (M00(k) * dj1Mtemp00) + cTmp4
        dJ01Mtemp01 = (M00(k) * dj1Mtemp01) - cTmp4
        cTmp4 = M11(k) * dj1Mtemp10
        dJ01Mtemp10 = (M10(k) * dj1Mtemp00) + cTmp4
        dJ01Mtemp11 = (M10(k) * dj1Mtemp01) - cTmp4

        dJ00(k) = dudsig * (dJ_10Mtemp00 + dJ01Mtemp00)
        dJ10(k) = dudsig * (dJ_10Mtemp10 + dJ01Mtemp10)
        dJ01(k) = dudsig * (dJ_10Mtemp01 + dJ01Mtemp01)
        dJ11(k) = dudsig * (dJ_10Mtemp11 + dJ01Mtemp11)
      enddo

      ! k = n_layer
      utemp0 = utemp1
      c1t = c1(nLayers - 1)
      c2t = c1(nLayers)
      dudsig = 0.5d0 * cTmp2 * c2t / utemp0

      h_1 = thickness(nLayers - 1)

      dJ_10Mtemp00 = sqrt(lam2 + cTmp2 * c1t * sig(nLayers - 1, i, jj))
      dJ_10Mtemp11 = c1t / (c2t * dJ_10Mtemp00)
      dJ_10Mtemp01 = exp(-2.d0 * dJ_10Mtemp00 * h_1)

      dj0Mtemp11 =  0.5d0 * dJ_10Mtemp11
      dj0Mtemp00 =  dj0Mtemp11 * dJ_10Mtemp01

      cTmp4 = dudsig * dj0Mtemp00
      dJ00(nLayers) = cTmp4
      dJ01(nLayers) = -cTmp4
      cTmp4 = dudsig * dj0Mtemp11
      dJ10(nLayers) = -cTmp4
      dJ11(nLayers) = cTmp4

      ! Second pass, Double loop
      ! k1 = 1
      dJ1sum00 = (0.d0, 0.d0) ; dJ1sum10 = (0.d0, 0.d0)
      dJ1sum01 = (0.d0, 0.d0) ; dJ1sum11 = (0.d0, 0.d0)

      dJ0sum00 = dJ1sum00 ; dJ0sum10 = dJ1sum10
      dJ0sum01 = dJ1sum01 ; dJ0sum11 = dJ1sum11

      ! k = 1
      if (nLayers > 2) then
        dJ1sum00 = (dJ00(1) * M00(3)) + (dJ01(1) * M10(3))
        dJ1sum10 = (dJ10(1) * M00(3)) + (dJ11(1) * M10(3))
        dJ1sum01 = (dJ00(1) * M01(3)) + (dJ01(1) * M11(3))
        dJ1sum11 = (dJ10(1) * M01(3)) + (dJ11(1) * M11(3))

        dJ0sum00 = dJ1sum00 ; dJ0sum10 = dJ1sum10
        dJ0sum01 = dJ1sum01 ; dJ0sum11 = dJ1sum11

        do k = 2, nLayers - 2
          dJ1sum00 = (dJ0sum00 * M00(k + 2)) + (dJ0sum01 * M10(k + 2))
          dJ1sum10 = (dJ0sum10 * M00(k + 2)) + (dJ0sum11 * M10(k + 2))
          dJ1sum01 = (dJ0sum00 * M01(k + 2)) + (dJ0sum01 * M11(k + 2))
          dJ1sum11 = (dJ0sum10 * M01(k + 2)) + (dJ0sum11 * M11(k + 2))

          dJ0sum00 = dJ1sum00 ; dJ0sum10 = dJ1sum10
          dJ0sum01 = dJ1sum01 ; dJ0sum11 = dJ1sum11
        enddo
      endif

      cTmp3 = 1.d0 / M1sum11**2.d0

      drTE(1, i, jj) = (dJ1sum01 * M1sum11 - dJ1sum11 * M1sum01) * cTmp3

      do k1 = 2, nLayers - 1
        dJ0sum00 = M00(1) ; dJ0sum10 = M10(1)
        dJ0sum01 = M01(1) ; dJ0sum11 = M11(1)

        do k = 1, k1 - 2
          dJ1sum00 = (dJ0sum00 * M00(k + 1)) + (dJ0sum01 * M10(k + 1))
          dJ1sum10 = (dJ0sum10 * M00(k + 1)) + (dJ0sum11 * M10(k + 1))
          dJ1sum01 = (dJ0sum00 * M01(k + 1)) + (dJ0sum01 * M11(k + 1))
          dJ1sum11 = (dJ0sum10 * M01(k + 1)) + (dJ0sum11 * M11(k + 1))

          dJ0sum00 = dJ1sum00 ; dJ0sum10 = dJ1sum10
          dJ0sum01 = dJ1sum01 ; dJ0sum11 = dJ1sum11
        enddo

        ! k = i - 1
        dJ1sum00 = (dJ0sum00 * dJ00(k1)) + (dJ0sum01 * dJ10(k1))
        dJ1sum10 = (dJ0sum10 * dJ00(k1)) + (dJ0sum11 * dJ10(k1))
        dJ1sum01 = (dJ0sum00 * dJ01(k1)) + (dJ0sum01 * dJ11(k1))
        dJ1sum11 = (dJ0sum10 * dJ01(k1)) + (dJ0sum11 * dJ11(k1))

        dJ0sum00 = dJ1sum00 ; dJ0sum10 = dJ1sum10
        dJ0sum01 = dJ1sum01 ; dJ0sum11 = dJ1sum11

        do k = k1, nLayers - 2
          dJ1sum00 = (dJ0sum00 * M00(k + 2)) + (dJ0sum01 * M10(k + 2))
          dJ1sum10 = (dJ0sum10 * M00(k + 2)) + (dJ0sum11 * M10(k + 2))
          dJ1sum01 = (dJ0sum00 * M01(k + 2)) + (dJ0sum01 * M11(k + 2))
          dJ1sum11 = (dJ0sum10 * M01(k + 2)) + (dJ0sum11 * M11(k + 2))

          dJ0sum00 = dJ1sum00 ; dJ0sum10 = dJ1sum10
          dJ0sum01 = dJ1sum01 ; dJ0sum11 = dJ1sum11
        enddo

        drTE(k1, i, jj) = (dJ1sum01 * M1sum11 - dJ1sum11 * M1sum01) * cTmp3
      enddo

      ! k1 = nLayers
      dJ0sum00 = M00(1) ; dJ0sum10 = M10(1)
      dJ0sum01 = M01(1) ; dJ0sum11 = M11(1)

      do k = 1, nLayers - 2
        dJ1sum00 = (dJ0sum00 * M00(k + 1)) + (dJ0sum01 * M10(k + 1))
        dJ1sum10 = (dJ0sum10 * M00(k + 1)) + (dJ0sum11 * M10(k + 1))
        dJ1sum01 = (dJ0sum00 * M01(k + 1)) + (dJ0sum01 * M11(k + 1))
        dJ1sum11 = (dJ0sum10 * M01(k + 1)) + (dJ0sum11 * M11(k + 1))

        dJ0sum00 = dJ1sum00 ; dJ0sum10 = dJ1sum10
        dJ0sum01 = dJ1sum01 ; dJ0sum11 = dJ1sum11
      enddo

      ! k = nLayers - 1
      dJ1sum00 = (dJ0sum00 * dJ00(nLayers)) + (dJ0sum01 * dJ10(nLayers))
      dJ1sum10 = (dJ0sum10 * dJ00(nLayers)) + (dJ0sum11 * dJ10(nLayers))
      dJ1sum01 = (dJ0sum00 * dJ01(nLayers)) + (dJ0sum01 * dJ11(nLayers))
      dJ1sum11 = (dJ0sum10 * dJ01(nLayers)) + (dJ0sum11 * dJ11(nLayers))

      drTE(nLayers, i, jj) = (dJ1sum01 * M1sum11 - dJ1sum11 * M1sum01) * cTmp3
    enddo ! i = 1, nFrequencies
  enddo !jj = 1, nFilter
  end subroutine
  !====================================================================!
end module
