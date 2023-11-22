"""
Created on Tue Sep  6 10:09:27 2022

@author: german
"""
import numpy as np
from scipy.optimize import fsolve
R = 8.31446261815324
# P (bar->Pa) T (K)


def Pv(An, T):
    """
    Parameters
    ----------
    An : array, required
        Array with 3 Antoine parameters [A,B,C].
    T : array, required. Can be a value or an array
        Temperature, K.

    Returns
    -------
    Pv : 
        Vapour pressure, K. If T is an array, returns an array of Pv

    """
    Pv = 10**(An[0]-An[1]/(T+An[2]))
    return Pv


def abVW(T, CritPoint):
    """


    Parameters
    ----------
    T : Temperature, required

    CritPoint : array, required
        Substance parameters [TC,PC,FA,M (optional)].

    Returns
    -------
    [a,b]
        a,b parameters from VW equation.

    """
    [TC, PC] = CritPoint[0:2]
    a = 27/64*(R*TC)**2/PC
    b = 1/8*R*TC/PC
    return [a, b]


def VW(T, CritPoint, P=0, V=0):
    """
    Parameters
    ----------
    T : Temperature, real, required
        DESCRIPTION. The value of T (K).
    CritPoint : TYPE, required
        Substance parameters [TC,PC,FA,M]
    P : array, optional
        Pressure, Pa. The default is 0, if this value is known, the 3 values of 
        V are calculated 
    V : array, optional
        Molar volume, m3/mol. The default is 0, if this value is known, 
        the P is calculated


    Returns
    -------
    array
        Returns V(m3/mol) or P(Pa) calculate with VW equation.

    """
    [a, b] = abVW(T, CritPoint)
    if P == 0:
        if V == 0:
            print('Debe especificar P o V')
            return
        else:
            PVW = R*T/(V-b)-a/V**2
            return PVW
    else:
        VVW = np.roots([P, -(b*P+R*T), a, -a*b])
        return VVW


# buscar el valor de Psat con VW
def PsatVW(T, CritPoint):
    """
    Parameters
    ----------
    T : float, required
        temperature, K
    CritPoint : array, required
        Substance parameters [TC,PC,FA,M]
    Returns
    -------
    Psat (Pa) calculate with VW equation
    """

    if T >= CritPoint[0]:
        print('ERROR: T debe ser menor que la temperatura crítica,',CritPoint[0],'K')
        return CritPoint[1]
    
    [a, b] = abVW(T, CritPoint)
      

    def PsatVW0(P):
        # si P no es real, se usa el primer valor y si es negativo, P=1e-5
        if not type(P) == float:
            P = P[0]
        if P < 0:
            P = 1e-5
        Veq = VW(T=T, CritPoint=CritPoint, P=P)
        VL = min(Veq)
        VV = max(Veq)
        err = P*(VV-VL)-R*T*np.log((VV-b)/(VL-b))-(a/VV-a/VL)
        return err

    Vextr = np.roots(np.array([R*T, -2*a, 4*a*b, -2*a*b**2]))
    Vmax = Vextr[0]
    Vmin = Vextr[1]
    Pmax = VW(V=Vmax, T=T, CritPoint=CritPoint)
    Pmin = VW(V=Vmin, T=T, CritPoint=CritPoint)
    PsatVW = fsolve(PsatVW0, (Pmin+Pmax)/2)
    if not type(PsatVW) == float:
        PsatVW = PsatVW[0]
    return PsatVW


def abRK(T, CritPoint):
    """


    Parameters
    ----------
    T : Temperature, required

    CritPoint : array, required
        Substance parameters [TC,PC,FA,M (optional)].

    Returns
    -------
    [a,b,alfa]
        a,b,alfa parameters from RK equation.

    """
    [TC, PC] = CritPoint[0:2]
    alfa = (TC/T)**0.5
    a = 1/(9*(2**(1/3)-1))*alfa*(R*TC)**2/PC
    b = (2**(1/3)-1)/3*R*TC/PC
    return [a, b, alfa]


def RK(T, CritPoint, P=0, V=0):
    """
    Parameters
    ----------
    T : Temperature, real, required
        DESCRIPTION. The value of T (K).
    CritPoint : TYPE, required
        Substance parameters [TC,PC,FA,M]
    P : array, optional
        Pressure, Pa. The default is 0, if this value is known, the 3 values of 
        V are calculated 
    V : array, optional
        Molar volume, m3/mol. The default is 0, if this value is known, 
        the P is calculated


    Returns
    -------
    TYPE
        Returns V(m3/mol) or P(Pa) calculate with RK equation.

    """
    [a, b, alfa] = abRK(T, CritPoint)
    if P == 0:
        PRK = R*T/(V-b)-a/V/(V+b)
        return PRK
    else:
        VRK = np.roots([P, -R*T, a-b**2*P-R*T*b, -a*b])
        return VRK


# buscar el valor de Psat con RK
def PsatRK(T, CritPoint):
    """
    Parameters
    ----------
    T : float, required
        temperature, K
    CritPoint : array, required
        Substance parameters [TC,PC,FA,M]
    Returns
    -------
    Psat (Pa) calculate with RK equation
    """

    if T >= CritPoint[0]:
        print('ERROR: T debe ser menor que la temperatura crítica,',CritPoint[0],'K')
        return CritPoint[1]
    
    [a, b, alfa] = abRK(T, CritPoint)

    def PsatRK0(P):
        # si P es negativo, P=1e-5
        if not type(P) == float:
            P = P[0]
        if P < 0:
            P = 1e-5
        Veq = RK(P=P, T=T, CritPoint=CritPoint)
        VL = min(Veq)
        VV = max(Veq)
        err = P*(VV-VL)-R*T*np.log((VV-b)/(VL-b)) + \
            a/b*np.log(VV*(VL+b)/VL/(VV+b))
        return err
    # Las dos primeras soluciones de dP/dV=0 corresponden al mín. y max [0:2]
    Vextr = np.roots(
        np.array([R*T, 2*R*T*b-2*a, 3*a*b+R*T*b**2, 0, -a*b**3]))[0:2]
    # Calcula la presión en Vextr y si en el mín. es negativa, la hace 1e-5
    Pextr = [p if p > 0 else 1e-5 for p in RK(T, V=Vextr)]
    [PsatRK] = fsolve(PsatRK0, sum(Pextr)/2)
    # Los corchetes para que devuelva una variable tipo float
    return PsatRK


def abSRK(T, CritPoint):
    """


    Parameters
    ----------
    T : Temperature, required

    CritPoint : array, required
        Substance parameters [TC,PC,FA,M (optional)].

    Returns
    -------
    [a,b,alfa,m]
        a,b parameters from SRK equation.

    """
    [TC, PC, FA] = CritPoint[0:3]
    m = 0.48+1.574*FA-0.176*FA**2
    alfa = (1+m*(1-(T/TC)**0.5))**2
    a = 1/(9*(2**(1/3)-1))*alfa*R**2*TC**2/PC
    b = (2**(1/3)-1)/3*R*TC/PC
    return [a, b, alfa, m]


def SRK(T, CritPoint, P=0, V=0):
    """
    Parameters
    ----------
    T : Temperature, real, required
        DESCRIPTION. The value of T (K).
    CritPoint : TYPE, required
        Substance parameters [TC,PC,FA,M]
    P : array, optional
        Pressure, Pa. The default is 0, if this value is known, the 3 values of 
        V are calculated 
    V : array, optional
        Molar volume, m3/mol. The default is 0, if this value is known, 
        the P is calculated


    Returns
    -------
    TYPE
        Returns V(m3/mol) or P(Pa) calculate with SRK equation.

    """
    [a, b, alfa, m] = abSRK(T, CritPoint)
    if P == 0:
        PSRK = R*T/(V-b)-a/V/(V+b)
        return PSRK
    else:
        VSRK = np.roots([P, -R*T, a-b**2*P-R*T*b, -a*b])
        return VSRK

# buscar el valor de Psat con SRK


def PsatSRK(T, CritPoint):
    """
    Parameters
    ----------
    T : float, required
        temperature, K
    CritPoint : array, required
        Substance parameters [TC,PC,FA,M]
    Returns
    -------
    Psat (Pa) calculate with SRK equation
    """

    if T >= CritPoint[0]:
        print('ERROR: T debe ser menor que la temperatura crítica,',CritPoint[0],'K')
        return CritPoint[1]
    
    [a, b, alfa, m] = abSRK(T, CritPoint)

    def PsatSRK0(P):
        # si P es negativo, P=1e-5
        if not type(P) == float:
            P = P[0]
        if P < 0:
            P = 1e-5
        V_fases = SRK(T, CritPoint=CritPoint, P=P)
        VL = min(V_fases)
        VV = max(V_fases)
        err = P*(VV-VL)-R*T*np.log((VV-b)/(VL-b)) + \
            a/b*np.log(VV*(VL+b)/VL/(VV+b))
        if err.imag == 0:
            return err
        else:
            return 1e3

    # Las dos primeras soluciones de dP/dV=0 corresponden al mínimo y máximo
    Vextr = np.roots(
        np.array([R*T, 2*R*T*b-2*a, 3*a*b+R*T*b**2, 0, -a*b**3]))[0:2]
    Pextr = [p if p > 0 else 1e-5 for p in SRK(T, V=Vextr)]
    [PsatSRK] = fsolve(PsatSRK0, sum(Pextr)/2)
    return PsatSRK


def abPR(T, CritPoint):
    """


    Parameters
    ----------
    T : Temperature, required

    CritPoint : array, required
        Substance parameters [TC,PC,FA,M (optional)].

    Returns
    -------
    [a,b,alfa,m]
        a,b parameters from PR equation.

    """
    [TC, PC, FA] = CritPoint[0:3]
    # Values of https://en.wikipedia.org/wiki/Cubic_equations_of_state
    ZC = 1/32*(11-2*7**0.5*np.sinh(1/3*np.arcsinh(13/7/7**0.5)))
    bb = 1/3*(8**0.5*np.sinh(1/3*np.arcsinh(8**0.5))-1)*ZC
    b = bb*R*TC/PC
    m = 0.37464+1.54226*FA-0.26992*FA**2
    alfa = (1+m*(1-(T/TC)**0.5))**2
    aa = 8/3/(1+np.cosh(1/3*np.arccosh(3)))*ZC**3/bb
    a = aa*alfa*R**2*TC**2/PC
    return [a, b, alfa, m]


def PR(T, CritPoint, P=0, V=0):
    """
    Parameters
    ----------
    T : Temperature, real, required
        DESCRIPTION. The value of T (K).
    CritPoint : TYPE, required
        Substance parameters [TC,PC,FA,M]
    P : array, optional
        Pressure, Pa. The default is 0, if this value is known, the 3 values of 
        V are calculated 
    V : array, optional
        Molar volume, m3/mol. The default is 0, if this value is known, 
        the P is calculated


    Returns
    -------
    TYPE
        Returns V(m3/mol) or P(Pa) calculate with PR equation.

    """
    [a, b, alfa, m] = abPR(T, CritPoint)
    if P == 0:
        PPR = R*T/(V-b)-a/(V*(V+b)+b*(V-b))
        return PPR
    else:
        VPR = np.roots([P, P*b-R*T, a-3*P*b**2-2*R*T*b, b**3*P+b**2*R*T-a*b])
        return VPR

# buscar el valor de Psat con PR


def PsatPR(T, CritPoint):
    """
    Parameters
    ----------
    T : float, required
        temperature, K
    CritPoint : array, required
        Substance parameters [TC,PC,FA,M]
    Returns
    -------
    Psat (Pa) calculate with PR equation
    """

    if T >= CritPoint[0]:
        print('ERROR: T debe ser menor que la temperatura crítica,',CritPoint[0],'K')
        return CritPoint[1]
    
    [a, b, alfa, m] = abPR(T, CritPoint)

    def PsatPR0(P):
        # si P es negativo, P=1e-5
        if not type(P) == float:
            P = P[0]
        if P < 0:
            P = 1e-5
        V_fases = PR(T, CritPoint=CritPoint, P=P)
        VL = min(V_fases)
        VV = max(V_fases)
        b1 = -b*(1-2**0.5)
        b2 = -b*(1+2**0.5)
        err = P*(VV-VL)-R*T*np.log((VV-b)/(VL-b))+a/2**1.5 / \
            b*np.log((VV-b1)*(VL-b2)/(VL-b1)/(VV-b2))
        if err.imag == 0:
            return err
        else:
            return 1e3

    Vextr = np.roots(np.array(
        [R*T, 4*b*R*T-2*a, 2*b**2*R*T+2*a*b, 2*a*b**2-4*b**3*R*T, b**4*R*T-2*a*b**3]))
    Vextr[::-1].sort()  # Es necesario ordenar las raíces de mayor a menor
    Pextr = [p if p > 0 else 1e-5 for p in PR(T, CritPoint=CritPoint, V=Vextr[0:2])]
    # [ ] Para evitar que la salida sea un array
    [PsatPR] = fsolve(PsatPR0, sum(Pextr)/2)
    return PsatPR
