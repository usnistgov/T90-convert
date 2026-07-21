import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from functools import lru_cache

def t90(Tin: float, scale: str):
    """
    Convert temperature from specified scale to ITS-90.
    Parameters:
    ----------
    Tin : float
        Temperature in the input scale (K except for 'C', 'F', and 'R' scales)
    scale : str specifying scale; choices are: 
        'C' (or 'Celsius'): Celsius temperature, assumed to be ITS-90, degC
        'F' (or 'Fahrenheit'): Fahrenheit temperature, assumed to be ITS-90, degF
        'R' (or 'Rankine'): Rankine temperature, assumed to be ITS-90, degR
        '68' (or '1968' or 'IPTS-68'): International Practical Temperature Scale of 1968
        'NBS68' (or 'NBS-IPTS-68' or 'NBS-68'): NBS implementation of IPTS-68
        '48' (or '1948' or 'ITS-48' or 'IPTS-48'): International (Practical) Temperature Scale of 1948
        '27' (or '1927' or 'ITS-27'): International Temperature Scale of 1927
        'EN' (or 'NHS'): Echelle Normale (Normal Hydrogen Scale) from 1887
        'PLTS' (or 'PLTS-2000' or 'PLTS2000'): Provisional Low Temperature Scale of 2000
        'PTB2006' (or 'PTB-2006'): 3He vapor-pressure scale of Engert et al., Metrologia 44, 40 (2007)
        '76' (or 'EPT76' or 'EPT-76'): 1976 Provisional 0.5 K to 30 K Temperature Scale
        'NPL75' (or 'NPL-75'): 1975 NPL gas thermometry scale 
        'XAc' (or 'ISUmag'): XAc' magnetic thermometry scale from ISU
        'm(III)' (or 'M(III)' or 'mIII' or 'KOLmag'): T_m(III) magnetic thermometry scale from KOL
        'XNML': XNML magnetic thermometry scale from NML
        'MAS': MAS magnetic thermometry scale from NML
        'XPRMI' (or 'PRMImag'): XPRMI magnetic thermometry scale from PRMI
        'X1' (or 'NPLmag'): X1 magnetic thermometry scale from NPL
        'NBS55' (or 'NBS-55' or '55'): 1955 NBS cryogenic termperature scale
        'NBS39' (or 'NBS-39' or '39'): 1937 NBS cryogenic temperature scale
        'NPL61' (or 'NPL-61'): 1961 NPL cryogenic temperature scale
        'PRMI54' (or 'PRMI-54'): 1954 PRMI cryogenic temperature scale
        'PSU54' (or 'PSU-54'): 1954 PSU cryogenic temperature scale
        'NBS220' (or 'NBS2-20' or 'NBS 2-20'): NBS Provisional Temperature Scale 2-20 (acoustic)
        'Michels' (or 'VDW' or 'Amsterdam'): Temperature scale used by Michels group at the van der Waals Laboratory in Amsterdam
        'Giauque' (or 'CAL'): Scale of Giauque lab at University of California, Berkeley
        'PTR': Scale of Physikalisch-Technische Reichsanstalt (PTR), Berlin
        'KOL' (or 'Keesom'): Scale of Keesom lab at Kamerlingh Onnes Laboratory, Leiden
        'GL' (or 'Carnegie'): Scale of Geophysical Laboratory and Carnegie Institution
        '58He' (or 'He58' or '58'): 1958 helium vapor pressure scale
        '62He' (or 'He62' or '62'): 1962 He-3 vapor pressure scale
        '55E': 55E helium vapor pressure scale of Clement (NRL)
        'L55': L55 helium vapor pressure scale of van Dijk (Leiden)
        'He48' (or '48He'): 1948 helium vapor pressure scale of van Dijk (Leiden)
        'He37' (or '37He'): 1937 helium vapor pressure scale of Schmidt and Keesom (Leiden)
        'He32' (or '32He'): 1932 helium vapor pressure scale of Keesom (Leiden)
        'He29' (or '29He'): 1929 helium vapor pressure scale of Keesom et al. (Leiden)
        'He24' (or '24He'): 1924 helium vapor pressure scale of Verschaffelt and Kamerlingh Onnes & Weber (Leiden)
        'HeBS' (or 'BS' or 'He39' or '39He'): 1939 helium vapor pressure scale of Bleaney and Simon

    Returns:
    ----------        
    T90 : float
        Temperature on ITS-90 scale (K)
    """
    # Strip leading and trailing blanks from scale
    scale = scale.strip()
    try:
        match scale:
            case 'C' | 'Celsius':
                return t_from_C(Tin)
            case 'F' | 'Fahrenheit':
                return t_from_F(Tin)
            case 'R' | 'Rankine':
                return t_from_R(Tin)
            case '68' | '1968' | 'IPTS-68':
                return t90_from_68(Tin)
            case 'NBS68' | 'NBS-IPTS-68'| 'NBS-68':
                return t90_from_NBS68(Tin)
            case '48' | '1948' | 'ITS-48' | 'IPTS-48':
                return t90_from_48(Tin)
            case '27' | '1927' | 'ITS-27':
                return t90_from_27(Tin)
            case 'EN' | 'NHS':
                return t90_from_EN(Tin)
            case 'PLTS' | 'PLTS-2000' | 'PLTS2000':
                return t90_from_PLTS(Tin)
            case 'PTB2006' | 'PTB-2006':
                return t90_from_PTB2006(Tin)
            case '76' | 'EPT76' | 'EPT-76':
                return t90_from_76(Tin)
            case 'NPL75' | 'NPL-75':
                return t90_from_NPL75(Tin)
            case 'XAc' | 'ISUmag':
                return t90_from_XAc(Tin)
            case 'm(III)' | 'M(III)' | 'mIII' | 'KOLmag':
                return t90_from_mIII(Tin)
            case 'XNML':
                return t90_from_XNML(Tin)
            case 'MAS':
                return t90_from_MAS(Tin)
            case 'XPRMI' | 'PRMImag':
                return t90_from_XPRMI(Tin)
            case 'X1' | 'NPLmag':
                return t90_from_X1(Tin)
            case 'NBS55' | 'NBS-55' | '55':
                return t90_from_NBS55(Tin)
            case 'NBS39' | 'NBS-39' | '39':
                return t90_from_NBS39(Tin)
            case 'NPL61' | 'NPL-61':
                return t90_from_NPL61(Tin)
            case 'PRMI54' | 'PRMI-54':
                return t90_from_PRMI54(Tin)
            case 'PSU54' | 'PSU-54':
                return t90_from_PSU54(Tin)
            case 'NBS220' | 'NBS2-20' | 'NBS 2-20':
                return t90_from_NBS220(Tin)
            case 'Michels' | 'VDW' | 'Amsterdam':
                return t90_from_Michels(Tin)
            case 'Giauque' | 'CAL':
                return t90_from_Giauque(Tin)
            case 'PTR':
                return t90_from_PTR(Tin)
            case 'KOL' | 'Keesom':
                return t90_from_KOL(Tin)
            case 'GL' | 'Carnegie':
                return t90_from_GL(Tin)
            case '58He' | 'He58' |'58':
                return t90_from_58(Tin)
            case '62He' | 'He62' | '62':
                return t90_from_62(Tin)
            case '55E' | 'He55E' :
                return t90_from_55E(Tin)
            case 'L55' | 'HeL55':
                return t90_from_L55(Tin)
            case 'He48' | '48He':
                return t90_from_He48(Tin)
            case 'He37' | '37He':
                return t90_from_He37(Tin)
            case 'He32' | '32He':
                return t90_from_He32(Tin)
            case 'He29' | '29He':
                return t90_from_He29(Tin)
            case 'He24' | '24He':
                return t90_from_He24(Tin)
            case 'HeBS' | 'BS' | 'He39' | '39He':
                return t90_from_HeBS(Tin)
            case _:
                raise ValueError(f"Scale '{scale}' not recognized")
    except ValueError as e:
        raise ValueError(f"Error converting temperature: {e}")

def t_from_C(tC):
    """
    Convert temperature from Celsius to kelvins
    
    Units: 
    - tC: Celsius temperature
    - Output: K
    """
    return tC + 273.15

def t_from_F(tF):
    """
    Convert temperature from Fahrenheit to kelvins
    
    Units: 
    - tF: Fahrenheit temperature
    - Output: K
    """
    tC = (tF - 32) / 1.8
    return tC + 273.15

def t_from_R(tR):
    """
    Convert temperature from Rankine to kelvins
    
    Units: 
    - tR: Rankine temperature
    - Output: K
    """
    return tR / 1.8

def t90_from_68(T68):
    """
    Convert temperature from IPTS-68 to ITS-90
    Note that lower limit of IPTS-68 was 13.81 K, so raise ValueError if T68 below that
    Uses equations in R.L. Rusby, "The conversion of thermal reference values to the ITS-90,"
    J. Chem. Thermodyn. 23, 1153 (1991)
    Conversion between roughly 630 and 1064 degC corrected according to Rusby et al.,
    Metrologia 31, 149 (1994)

    Parameters:
    ----------
    T68 : float
        Temperature on IPTS-68 scale (K)

    Returns:
    -------
    T90 : float
        Temperature on ITS-90 scale (K)
    """
    Tmin = 13.81
    if T68 < Tmin:
        raise ValueError("T68 cannot be below 13.81 K")
    elif T68 < 77.0:
        ai = np.array([-0.005903, 0.008174, -0.061924, -0.193388, 1.490793, 
                       1.252347, -9.835868, 1.411912, 25.277595, -19.183815, 
                       -18.437089, 27.000895, -8.716324])
        frac = (T68 - 40.0) / 40.0
        poly = Polynomial(ai)
        return T68 + poly(frac)
    elif T68 < 903.89:
        bi = np.array([0.0, -0.148759, -0.267408, 1.080760, 1.269056, 
                       -4.089591, -1.871251, 7.438081, -3.536296])
        frac = (T68 - 273.15) / 630.0
        poly = Polynomial(bi)
        return T68 + poly(frac)
    elif T68 < 1337.58:
# Note that equation for this range uses t90 as independent variable, 
# so we need to solve iteratively. Start with T68 as initial guess for T90 and iterate until convergence
        T_new = T68
        ci = np.array([7.8687209e1, -4.7135991e-1, 1.0954715e-3,
                       -1.2357884e-6, 6.7736583e-10, -1.4458081e-13])
        poly = Polynomial(ci)
        while True:
            tc = T_new - 273.15
            T_old = T_new
            T_new = T68 + poly(tc)
            if abs(T_new - T_old) < 1e-11:
                return T_new
    else:
        diff = -0.25 * (T68/1337.58)**2
        return T68 + diff

def t90_from_NBS68(TNBS):
    """
    Convert temperature from NBS-IPTS-68 to ITS-90
    Note that lower limit of IPTS-68 was 13.81 K, so raise ValueError if T68 below that
    Uses differences between NBS-IPTS-68 and NPL IPTS-68 from Ward and Compton (1979)
    as provided in personal communication from W. Tew (2026)
    Above 273.15 K scale is identical to IPTS-68

    Parameters:
    ----------
    TNBS : float
        Temperature on NBS-IPTS-68 scale (K)

    Returns:
    -------
    T90 : float
        Temperature on ITS-90 scale (K)
    """
    Tmin = 13.81
    if TNBS < Tmin:
        raise ValueError("T68 cannot be below 13.81 K")
    elif TNBS < 273.15:
        T_data = np.array([
        13.81, 14.006, 14.506, 15.003, 15.5035, 16.004, 16.50515, 17.042, 17.50715, 
        18.008, 18.5085, 19.009, 19.509, 20.28, 21.008, 22.007, 23.007, 24.006, 
        24.5616, 25.005, 26.0045, 27.1041, 28.005, 30.006, 32.008, 34.008, 36.007, 
        38.006, 40.006, 45.007, 50.006, 54.361, 59.997, 64.995, 69.993, 74.992, 
        79.992, 83.798, 90.192, 99.991, 119.987, 139.986, 159.986, 179.988, 
        199.989, 219.991, 234.3082, 239.993, 259.997, 273.15
        ])
        Diff_data = np.array([
        -4.0, -2.9, -1.0, -0.2, -0.1, -0.4, -0.75, -1.3, -1.8, -2.0, -2.3, -2.5, 
        -2.3, -1.8, -1.0, -0.2, 0.35, 0.75, 1.2, 1.0, 1.1, 1.2, 1.2, 1.1, 0.9, 
        0.6, 0.4, 0.2, -0.1, -0.3, -0.1, 0.6, 1.7, 2.4, 2.9, 3.25, 3.4, 3.4, 
        3.1, 2.5, 1.6, 0.9, 0.4, 0.0, -0.1, -0.15, -0.18, -0.2, -0.1, 0.0
        ])

        diff = 1.e-3 * np.interp(TNBS, T_data, Diff_data)
        T68 = TNBS - diff
        return t90_from_68(T68)
    else:
        return t90_from_68(TNBS)

def t90_from_48(T48):
    """
    Convert temperature from ITS-48 to ITS-90
    Note that lower limit of ITS-48 was 90.18 K, so raise ValueError if T48 below that
    Converts to T68 first (from Douglas, 1969) and then to T90 using T68to90 function
    which uses Rusby (1991 and 1994).
    For T between 51.15 K and 90.18 K, this yields the extended 1948 scale, IPTS-48*,
    defined by Lovejoy, Nature 197, 353 (1963) from whence it can be converted to NBS-55

    Parameters:
    ----------
    T48 : float
        Temperature on ITS-48 scale (K)

    Returns:
    -------
    T90 : float
        Temperature on ITS-90 scale (K)
    """
    Tmin2 = 51.15
    Tmin = 90.18
    if T48 < Tmin2:
        raise ValueError("T48 cannot be below 51.15 K")
    elif T48 > 10000:
        raise ValueError("T48 cannot be above 10000 K")
    elif T48 < Tmin:
        T55 = t55_from_48star(T48)
        T90 = t90_from_NBS55(T55)
        return T90
    else:
        T68 = t68_from_48(T48)
        T90 = t90_from_68(T68)
        return T90

def t90_from_27(T27):
    """
    Convert temperature from ITS-27 to ITS-90
    Note that lower limit of ITS-48 was 90.18 K, so raise ValueError if T27 below that
    Converts to T48 first (from Corruccini, 1949) and then to T90 using t90_from48 function.

    Parameters:
    ----------
    T27 : float
        Temperature on ITS-27 scale (K)

    Returns:
    -------
    T90 : float
        Temperature on ITS-90 scale (K)
    """
    Tmin = 90.18
    Tmax = 4522.15 # Maximum T for which T27 -> T48 conversion is valid
    if T27 < Tmin:
        raise ValueError(f"T27 cannot be below {Tmin} K")
    elif T27 > Tmax:
        raise ValueError(f"T27 cannot be above {Tmax} K")
    else:
        T48 = t48_from_27(T27)
        T90 = t90_from_48(T48)
        return T90
    
def t90_from_EN(TEN):
    """
    Convert temperature from Echelle Normale to ITS-90
    Lower limit is mercury freezing point. Per Hall (1930), negligibly different from ITS-27 between 0 and 100 degC.
    Assumes same as ITS-27 between 0 and 100 degC, linearly interpolates to difference at Hg freezing point.

    Parameters:
    ----------
    TEN : float
        Temperature on Echelle normale scale (K)

    Returns:
    -------
    T90 : float
        Temperature on ITS-90 scale (K)
    """
    TC = TEN - 273.15
    if -39.286 <= TC <= 0:
        TC_points = np.array([-39.286, 0.])
        diff_points = np.array([0.457, 0.])
        diff = np.interp(TC, TC_points, diff_points)
        TC90 = TC + diff
        return TC90 + 273.15
    elif 0 < TC <= 100:
        return t90_from_27(TEN)
    else:
        raise ValueError("Echelle normale temperature must be between -39.286 and 100 degrees C.")

def t68_from_48(T48):
    """
    Converts ITS-48 temperatures to IPTS-68.
    Based on Douglas, J. Res. NBS 73A, 451 (1969).

    Parameters:
    ----------
    T48 : float
        Temperature on ITS-48 scale (K)

    Returns:
    -------
    T68 : float
        Temperature on IPTS-68 scale (K)
    """
    T_ice = 273.15
    
    # Range check 
    if T48 < 90.18 or T48 > 10000.0:
        raise ValueError(f"ITS-48 Temperature {T48} K is out of range (90.18 - 10000 K).")

    T_new = T48
    # Case 1: 90.18 to 273.15 K
    if T48 < T_ice:
        while True:
            w_term = w68(T_new)
            tc = T_new - T_ice
            t1 = 3.984517e-3 * tc
            t2 = 5.855019e-7 * (tc**2)
            t3 = 4.35717e-12 * (100.0 - tc) * (tc**3)
            
            top = 250.97 * (1.0 + t1 - t2 + t3 - w_term)
            bot = 1.0 - 2.9389e-4 * tc + 4.3741e-9 * (75.0 - tc) * (tc**2)
            
            diff = top / bot
            T_old = T_new
            T_new = T48 + diff
            if abs(T_new - T_old) < 1e-12:
                return T_new

    # Case 2: 273.15 to 903.688 K
    elif T48 < 903.688:
        while True:
            tc = T_new - T_ice
            phi = 0.045 * (1e-2 * tc) * (1e-2 * tc - 1.0) * (tc / 419.58 - 1.0) * (tc / 630.74 - 1.0)
            top = 4.904e-7 * tc * (tc - 100.0)
            bot = 1.0 - 2.939e-4 * tc
            
            diff = (top / bot) + phi
            T_old = T_new
            T_new = T48 + diff
            if abs(T_new - T_old) < 1e-12:
                return T_new

    # Case 3: 903.688 to 1336.15 K
    elif T48 < 1336.15:
        while True:
            tc = T_new - T_ice
            top = -1.3145 + 1.5016e-3 * tc + 1.5625e-6 * (tc**2)
            bot = 1.0 + 4.101e-4 * tc
            
            diff = top / bot
            T_old = T_new
            T_new = T48 + diff
            if abs(T_new - T_old) < 1e-12:
                return T_new
            
    # Case 4: Above 1336.15 K
    else:
        while True:
            diff = 5.56e-4*T_new + 3.84e-7*(1-np.exp(-22135/T_new))*T_new**2
            T_old = T_new
            T_new = T48 + diff
            if abs(T_new - T_old) < 1e-9:
                return T_new

def w68(T68):
    """
    Solve for IPTS-68 reference function W_CCT-68 according to Metrologia Vol. 5, 35.
    Valid for T68 above 90 K and below 273.15 K.

    Parameters:
    ----------
    T68 : float
        Temperature on IPTS-68 scale (K)

    Returns:
    -------
    W68 : float
        Thermometry reference function W for 1968 temperature scale
    """
    # AI coefficients from the Fortran DATA statement
    # AI[0] is the offset, AI[1:21] are the polynomial coefficients
    AI = [
        273.15, 250.8462096788033, 135.0998699649997,
        52.78567590085172, 27.67685488541052, 39.10532053766837,
        65.56132305780693, 80.80358685598667, 70.52421182340520,
        44.78475896389657, 21.25256535560578, 7.679763581708458,
        2.136894593828500, 0.4598433489280693, 7.636146292316480e-2,
        9.693286203731213e-3, 9.230691540070075e-4, 
        6.381165909526538e-5, 3.022932378746192e-6, 
        8.775513913037602e-8, 1.177026131254774e-9
    ]

    # Initial boundaries
    whigh = 1.0
    Thigh = 273.15
    Tlow = 90.0
    wlow = 0.24298315
    
    # Maximum iterations to prevent infinite loops (safety measure)
    for _ in range(1000):
        # Linear interpolation to find the next W candidate
        delt = Thigh - Tlow
        delw = whigh - wlow
        frac = (T68 - Tlow) / delt
        wnew = wlow + frac * delw
        
        # Calculate Tnew based on the IPTS-68 polynomial
        wln = np.log(wnew)
        sum_val = 0.0
        for i in range(1, 21):
            sum_val += AI[i] * (wln ** i)
        
        Tnew = AI[0] + sum_val
        
        # Check for convergence
        if abs(Tnew - T68) < 1.0e-12:
            return wnew
        
        # Update boundaries (Bisection/False Position logic)
        if Tnew > T68:
            Thigh = Tnew
            whigh = wnew
        else:
            Tlow = Tnew
            wlow = wnew
            
    return wnew # Return best estimate if max iterations reached

@lru_cache(maxsize=1)
def get_2748_spline():
    """
    Cubic spline interpolation for converting ITS-27 to ITS-48 based on Table 2 in 
    R.J. Corruccini, J. Res. NBS 42, 133-136 (1949).
    """
    # Only used above 1063 degC, but add 3 lower points to better constrain behavior at low T
    # since more points are available and the difference at 960.5 degC is defined.
    x_1927 = np.array([
        960.50, 999.80, 1049.95, 1063.00, 1100.2, 1200.6, 1301.1, 1401.7, 1502.3, 1603.0, 
        1703.8, 1804.6, 1905.5, 2006.4, 2107, 2208, 2310, 2411, 2512, 2613, 
        2715, 2816, 2918, 3020, 3122, 3223, 3325, 3428, 3530, 3632, 3735, 
        3837, 3940, 4043, 4146, 4249
    ])

    y_1948 = np.array([
        960.80, 1000, 1050, 1063.00, 1100, 1200, 1300, 1400, 1500, 1600, 
        1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 
        2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 
        3800, 3900, 4000, 4100, 4200
    ])
    return CubicSpline(x_1927, y_1948)

def t48_from_27(T27):
    """
    Converts ITS-27 temperatures to ITS-48.
    Based on R.J. Corruccini, J. Res. NBS 42, 133-136 (1949).

    Parameters:
    ----------
    T27 : float
        Temperature on ITS-27 scale (K)

    Returns:
    -------
    T48 : float
        Temperature on ITS-48 scale (K)
    """
    
    # Range check 
    if T27 < 90.18 or T27 > (4249 + 273.15):
        raise ValueError(f"ITS-27 Temperature {T27} K is out of range (90.18 - 4522.15 K).")

    TC27 = T27 - 273.15
    
    # Case 1: No change below 630.50 degC
    if TC27 < 630.5:
        return T27

    elif TC27 < 1063:
    # In thermocouple range, use equation for derivative and explanation of 0.3 degree shift
    # given on p. 134 of Corruccini paper
        top = -226.466 + 0.5722281*TC27 - 0.0003378967*TC27*TC27
        bottom = 8.227 + 0.003309*TC27
        TC48 = TC27 + 0.3 * top / bottom
        return TC48 + 273.15
    
    else:
    # Case 3: interpolate Table 2 in paper
    # Generate the Cubic Spline Interpolation
        cs = get_2748_spline()
        TC48 = cs(TC27)
        return TC48 + 273.15

def t90_from_PLTS(T_PLTS):
    """
    Convert temperature from Provisional Low Temperature Scale PLTS-2000 to ITS-90
    PLTS-2000 is valid from 0.9 mK to 1 K, but restrict to range where ITS-90 is valid
    which is above 0.65 K.
    
    Units: 
    - T_PLTS: K
    - Output: K (ITS-90)
    """
#   Tlow is PLTS-2000 temperature corresponding to 0.65 K on ITS-90
    Tlow = 0.648
    if T_PLTS < Tlow or T_PLTS > 1:
        raise ValueError(f"PLTS-2000 temperature must be between {Tlow} and 1 K.")

    else:
    # Interpolate based on differences from Pan et al., Metrologia 58, 025005 (2021)
        T_val = np.array([0.6479, 0.7005, 0.7515, 0.7985, 0.8507, 0.8985, 0.9489, 0.9972]) 
        Diff_val = 0.001 * np.array([1.577, 1.245, 1.094, 1.026, 0.718, 0.706, 0.647, 0.281])
        Diff = np.interp(T_PLTS, T_val, Diff_val)
        return T_PLTS + Diff

def t90_from_PTB2006(T2006):
    """
    Convert temperature from PTB-2006 3He Scale to ITS-90
    Raise ValueError if below T that corresponds to 0.65 K on ITS-90.
    Matches vapor pressure from PTB-2006 formulation to that of ITS-90.
    
    Units: 
    - T2006: K
    - Output: K (ITS-90)
    """
    
    Tlow = 0.648475 # corresponding to 0.65 K on ITS-90
    Thigh = 3.199993 # corresponding to 3.2 K on ITS-90
    
    if T2006 < Tlow or T2006 > 3.2:
        raise ValueError(f"PTB-2006 temperature must be between {Tlow} and 3.2 K.") 
    elif T2006 <= Thigh:
        p2006 = p_PTB2006(T2006)
        return t_3He90(p2006) 
    else:
    # in tiny range where PTB-2006 produces T above 3.2 K on ITS-90, 
    # apply constant offset 
        offset = t90_from_PTB2006(Thigh) - Thigh
        return T2006 + offset 

def t90_from_76(T76):
    """
    Convert temperature from 1976 Provisional T Scale EPT-76 
    EPT-76 was valid from 0.5 K to 30 K, but lower limit here is 0.65 K since that is
    lower limit of ITS-90.
    
    Units: 
    - T76: K (EPT-76)
    - Output: K (ITS-90)
    """
 
    if T76 < 0.65 or T76 > 30:
        raise ValueError(f"T76 must be between 0.65 and 30 K.")
    elif T76 <= 27.1:
    # Use equation from R.L. Rusby, J. Chem. Thermodyn. 23, 1153 (1991)
        diff = -5.6e-6 * T76**2
        return T76 + diff
    else:
    # Above 27.1 K, EPT-76 is considered to be equal to IPTS-68
    # This introduces a discontinuity of about 0.8 mK at 27.1 K
        return t90_from_68(T76)    

def t90_from_NPL75(T75):
    """
    Convert temperature from 1975 NPL gas thermometry scale to ITS-90.
    Converts to EPT-76 based on Table 5 in Metrologia 15, 65 (1979) 
    and then to ITS-90.
    
    Units: 
    - T75: K (NPL-75)
    - Output: K (ITS-90)
    """
    Tmin = 2.6
    Tmax = 27.1
    a = 0
    b = 0
    c = -0.0056
    d = 0

    if Tmin <= T75 <= Tmax:
        diff_mK = a + b*T75 + c*T75**2 + d/T75
        T76 = T75 - diff_mK * 1e-3
        return t90_from_76(T76)
    else:
        raise ValueError(f"NPL-75 temperature must be between {Tmin} and {Tmax} K.")    

def t90_from_XAc(Tmag):
    """
    Convert temperature from XAc' magnetic thermometry scale (ISU) to ITS-90.
    Converts to EPT-76 based on Table 5 in Metrologia 15, 65 (1979) 
    and then to ITS-90.
    
    Units: 
    - Tmag: K
    - Output: K (ITS-90)
    """
    Tmin = 1.1
    Tmax = 30
    a = 0
    b = 0
    c = 0.0025
    d = 0

    if Tmin <= Tmag <= Tmax:
        diff_mK = a + b*Tmag + c*Tmag**2 + d/Tmag
        T76 = Tmag - diff_mK * 1e-3
        return t90_from_76(T76)
    else:
        raise ValueError(f"XAc' temperature must be between {Tmin} and {Tmax} K.")    

def t90_from_mIII(Tmag):
    """
    Convert temperature from m(III) magnetic thermometry scale of KOL to ITS-90.
    Converts to EPT-76 based on Table 5 in Metrologia 15, 65 (1979) 
    and then to ITS-90.
    
    Units: 
    - Tmag: K
    - Output: K (ITS-90)
    """
    Tmin = 2
    Tmax = 27
    a = -8.0
    b = 1.5
    c = -0.0413
    d = 8.3

    if Tmin <= Tmag <= Tmax:
        diff_mK = a + b*Tmag + c*Tmag**2 + d/Tmag
        T76 = Tmag - diff_mK * 1e-3
        return t90_from_76(T76)
    else:
        raise ValueError(f"m(III) temperature must be between {Tmin} and {Tmax} K.")    

def t90_from_XNML(Tmag):
    """
    Convert temperature from XNML magnetic thermometry scale of NML to ITS-90.
    Converts to EPT-76 based on Table 5 in Metrologia 15, 65 (1979) 
    and then to ITS-90.
    
    Units: 
    - Tmag: K
    - Output: K (ITS-90)
    """
    Tmin = 1.1
    Tmax = 30
    a = -1.5
    b = 0.41
    c = -0.0109
    d = 0

    if Tmin <= Tmag <= Tmax:
        diff_mK = a + b*Tmag + c*Tmag**2 + d/Tmag
        T76 = Tmag - diff_mK * 1e-3
        return t90_from_76(T76)
    else:
        raise ValueError(f"XNML temperature must be between {Tmin} and {Tmax} K.")   

def t90_from_MAS(Tmag):
    """
    Convert temperature from MAS magnetic thermometry scale of NML to ITS-90.
    Converts to EPT-76 based on Table 5 in Metrologia 15, 65 (1979) 
    and then to ITS-90.
    
    Units: 
    - Tmag: K
    - Output: K (ITS-90)
    """
    Tmin = 1.1
    Tmax = 30
    a = -1.5
    b = 0.49
    c = -0.0125
    d = 0

    if Tmin <= Tmag <= Tmax:
        diff_mK = a + b*Tmag + c*Tmag**2 + d/Tmag
        T76 = Tmag - diff_mK * 1e-3
        return t90_from_76(T76)
    else:
        raise ValueError(f"MAS temperature must be between {Tmin} and {Tmax} K.")    

def t90_from_XPRMI(Tmag):
    """
    Convert temperature from XPRMI magnetic thermometry scale of PRMI to ITS-90.
    Converts to EPT-76 based on Table 5 in Metrologia 15, 65 (1979) 
    and then to ITS-90.
    
    Units: 
    - Tmag: K
    - Output: K (ITS-90)
    """
    Tmin = 4.2
    Tmax = 27
    a = 0
    b = 0.51
    c = -0.0125
    d = 0

    if Tmin <= Tmag <= Tmax:
        diff_mK = a + b*Tmag + c*Tmag**2 + d/Tmag
        T76 = Tmag - diff_mK * 1e-3
        return t90_from_76(T76)
    else:
        raise ValueError(f"XPRMI temperature must be between {Tmin} and {Tmax} K.")    
    
def t90_from_X1(Tmag):
    """
    Convert temperature from X1 magnetic thermometry scale of NPL to ITS-90.
    Converts to EPT-76 based on Table 5 in Metrologia 15, 65 (1979) 
    and then to ITS-90.
    
    Units: 
    - Tmag: K
    - Output: K (ITS-90)
    """
    Tmin = 0.65
    Tmax = 3.1
    a = 0
    b = 0
    c = 0
    d = 0

    if Tmin <= Tmag <= Tmax:
        diff_mK = a + b*Tmag + c*Tmag**2 + d/Tmag
        T76 = Tmag - diff_mK * 1e-3
        return t90_from_76(T76)
    else:
        raise ValueError(f"NPL X1 temperature must be between {Tmin} and {Tmax} K.")    

def t90_from_NBS55(T55):
    """
    Calculates the ITS-90 temperature from the NBS-55 scale.
    First convert to IPTS-68, then to ITS-90.
    Lower limit is e-H2 triple point; scale was sometimes used at lower T
    but unable to find clear conversion to something like EPT-76.
    
    Units: 
    - T55: K (NBS-55)
    - Output: K (ITS-90)
    """
    if T55 < 13.8135 or T55 > 91:
        raise ValueError("NBS-55 temperature must be between 13.8135 and 91 K.")
    else:
        T68 = t68_from_NBS55(T55)
        return t90_from_68(T68)
    
def t90_from_NBS39(T39):
    """
    Calculates the ITS-90 temperature from the NBS-39 scale, which was
    given by Hoge & Brickwedde, J. Res. NBS 22, 351 (1939).
    This was identical to NBS-55 except for an increment of 0.01 K
    
    Units: 
    - T39: K (NBS-39)
    - Output: K (ITS-90)
    """
    if T39 < 13.8235 or T39 > 91.01:
        raise ValueError("NBS-39 temperature must be between 13.8235 and 91.01 K.")
    else:
        T55 = T39 - 0.01
        return t90_from_NBS55(T55)

def t90_from_NPL61(T61):
    """
    Calculates the ITS-90 temperature from the NPL-61 scale.
    First convert to IPTS-68, then to ITS-90.
    
    Units: 
    - T61: K
    - Output: K
    """
    if T61 < 13.8129 or T61 > 91:
        raise ValueError("NPL-61 temperature must be between 13.8129 and 91 K.")
    else:
        T68 = t68_from_NPL61(T61)
        return t90_from_68(T68)

def t90_from_PRMI54(T54):
    """
    Calculates the ITS-90 temperature from the PRMI-54 scale.
    First convert to IPTS-68, then to ITS-90.
    
    Units: 
    - T54: K
    - Output: K
    """
    if T54 < 13.859 or T54 > 91:
        raise ValueError("PRMI-54 temperature must be between 13.859 and 91 K.")
    else:
        T68 = t68_from_PRMI54(T54)
        return t90_from_68(T68)

def t90_from_PSU54(T54):
    """
    Calculates the ITS-90 temperature from the PSU-54 scale.
    First convert to IPTS-68, then to ITS-90.
    
    Units: 
    - T54: K
    - Output: K
    """
    if T54 < 13.801 or T54 > 91:
        raise ValueError("PSU-54 temperature must be between 13.801 and 91 K.")
    else:
        T68 = t68_from_PSU54(T54)
        return t90_from_68(T68)
    
def t90_from_NBS220(T220):
    """
    Calculates the ITS-90 temperature from the NBS 2-20 scale.
    NBS 2-20 is defined from 2.3 K to 20 K in H. Plumb and G. Cataland,
    Metrologia 2, 127-139 (1966)
    First convert to EPT-76, then to ITS-90
    
    Units: 
    - T220: K
    - Output: K
    """
    if T220 < 2.3 or T220 > 20.0:
        raise ValueError("NBS 2-20 temperature must be between 2.3 and 20 K.")
    else:
        T76 = t76_from_NBS220(T220)
        return t90_from_76(T76)

def t90_from_Michels(TMic):
    """
    Calculates the ITS-90 temperature from the values on the Michels T scale,
    used at the van der Waals Laboratory in Amsterdam, 
    Convert to 1948 scale according to Fig. 1 in Bedford & Prins, Physica 77, 121-125 (1974).
     
    Units: 
    - TMic: K (Michels lab scale)
    - Output: K (ITS-90)
    """
    if TMic < 83.145:
        raise ValueError("Michels T scale temperature must be above 83.145 K.")
    else:
        T48 = t48_from_Michels(TMic)
        return t90_from_48(T48)

def t90_from_Giauque(TG):
    """
    Calculates the ITS-90 temperature from the values on the scale
    used by Giauque at the university of California, Berkeley 
    Linear interpolation based on reported fixed points compared to ITS-90 values
    (either in ITS-90 definition or Bedford paper). 
    Points used are n-H2 TP and BP, O2 solid transition, N2 TP and BP, O2 TP and BP, CO2 subl. pt.
    ice point, water NBP. 
     
    Units: 
    - TG: K (Giauque lab scale)
    - Output: K (ITS-90)
    """
    if 13.92 <= TG <= 300:
        TG_points = np.array([13.92, 20.36, 43.76, 54.39, 63.14, 77.32, 90.13, 194.67, 273.1, 373.1])
        diff_points = np.array([0.032, 0.028, 0.036, -0.0316, 0.011, 0.032, 0.067, 0.016, 0.05, 0.0243])
        diff = np.interp(TG, TG_points, diff_points)
        return TG + diff
    else:
        raise ValueError("Giauque lab scale temperature must be between 13.92 and 300 K.")

def t90_from_PTR(TPTR):
    """
    Calculates the ITS-90 temperature from the values on the scale
    used at the PTR in Germany 
    Linear interpolation based on reported fixed points compared to ITS-90 values
    (either in ITS-90 definition or Bedford paper). 
    Points used are n-H2 TP and BP, NE TP and BP, O2 TP, N2 TP and BP, O2 BP, CO2 subl. pt.,
    Hg, freezing pt., ice point, water NBP. 
     
    Units: 
    - TPTR: K (PTR lab scale)
    - Output: K (ITS-90)
    """
    if 13.96 <= TPTR <= 373.16:
        TPTR_points = np.array([13.96, 20.38, 24.56, 27.073, 54.33, 63.14, 77.352, 90.19, 194.689, 234.328, 273.16, 373.16])
        diff_points = np.array([-0.008, 0.008, -0.0039, 0.026, 0.0284, 0.011, 0., 0.007, -0.003, -0.007, -0.01, -0.0357])
        diff = np.interp(TPTR, TPTR_points, diff_points)
        return TPTR + diff
    else:
        raise ValueError("PTR lab scale temperature must be between 13.96 and 373.16 K.")

def t90_from_KOL(TKOL):
    """
    Calculates the ITS-90 temperature from the values on the scale
    used at the Kamerlingh Onnes Laboratory in Leiden 
    Linear interpolation based on reported fixed points compared to ITS-90 values
    (either in ITS-90 definition or Bedford paper). 
    Points used are O2 BP, ice point, water NBP, sulfur BP, N2 BP, N2 TP. 
     
    Units: 
    - TKOL: K (KOL lab scale)
    - Output: K (ITS-90)
    """
    if 63.15 <= TKOL <= 717.714:
        TKOL_points = np.array([63.15, 77.357, 90.161, 273.144, 373.144, 717.714])
        diff_points = np.array([0.001, -0.005, 0.036, 0.006, -0.02, 0.05])
        diff = np.interp(TKOL, TKOL_points, diff_points)
        return TKOL + diff
    else:
        raise ValueError("KOL lab scale temperature must be between 63.15 and 717.714 K.")
    
def t90_from_GL(TGL):
    """
    Calculates the ITS-90 temperature from the values on the scale
    used at the Geophysical Laboratory of the Carnegie Institution 
    Linear interpolation based on reported fixed points compared to ITS-90 values
    (either in ITS-90 definition or Bedford paper). 
    Points used are ice point, water NBP, Sn MP, Cd MP, Zn MP, S BP, Sb MP, Ag MP,
    Au MP, Cu MP, Pd MP, Pt MP. 
     
    Units: 
    - TGL: K (GL lab scale)
    - Output: K (ITS-90)
    """
    if 273.15 <= TGL <= 2028.15:
        TGL_points = np.array([273.15, 373.15, 505.05, 594.05, 692.55, 717.70,
                                903.15, 1233.35, 1335.75, 1355.95, 
                                1822.65, 2028.15])
        diff_points = np.array([0.0, -0.026, 0.028, 0.169, 0.127, 0.064, 0.628,
                                1.58, 1.58, 1.82, 5.35, 13.15])
        diff = np.interp(TGL, TGL_points, diff_points)
        return TGL + diff
    else:
        raise ValueError("GL lab scale temperature must be between 273.15 and 2028.15 K.")
    
def t90_from_58(T58):
    """
    Convert temperature from 1958 He Scale to ITS-90
    Note that lower limit of 1958 He Scale was 0.5 K, so raise ValueError if T58 below that
    Matches vapor pressure from 1958 formulation to that of ITS-90
    Since ITS-90 4He equation only valid down to 1.25 K, switch to the 1962 3He scale
    (supposed to be equivalent) below that.
    For T above 5 K (on ITS-90) where the ITS-90 scale is not valid and the T58 upper limit of 5.22 K,
    apply the same difference as at the top of the range.
    
    Units: 
    - T58: K
    - Output: K (ITS-90)
    """
    Tlow = 0.64710571 # T at which ITS-90 is 0.65 K
    Tmid = 1.246806 # T at which ITS-90 is 1.25 K
    T5 = 4.9926976 # T corresponding to ITS-90 temperature of 5.0
    # Boundary check per the 1958 Scale limits (0.5 K to 5.22 K)
    if T58 < Tlow or T58 > 5.22:
        raise ValueError(f"T58 must be between {Tlow} and 5.22 K.")
    elif T58 < Tmid:
        return t90_from_62(T58)
    elif T58 <= T5:
        p58 = p_He58(T58)
        return t_He90(p58)
    else:
    # Above range of ITS-90 4He vapor pressure equation (but within T58 range),
    # apply constant offset since EPT-76 paper indicates the difference is constant above 4 K.
        return T58 + (t90_from_58(T5) - T5)

def t90_from_62(T62):
    """
    Convert temperature from 1962 3He Scale to ITS-90
    Raise ValueError if below T that corresponds to 0.65 K on ITS-90.
    Matches vapor pressure from 1962 formulation to that of ITS-90.
    If T above 3.2 K on ITS-90, switch to the 4He conversion.
    
    Units: 
    - T62: K
    - Output: K (ITS-90)
    """
    
    Tlow = 0.64710571 # corresponding to 0.65 K on ITS-90
    Thigh = 3.1936 # corresponding to 3.2 K on ITS-90
    
    if T62 < Tlow or T62 > 3.324:
        raise ValueError(f"T62 must be between {Tlow} and 3.324 K.") 
    elif T62 < Thigh:
        p62 = p_3He62(T62)
        return t_3He90(p62) 
    else:
        return t90_from_58(T62) # Supposed to be equivalent, so just use the 1958 4He conversion

def t90_from_55E(T55E):
    """
    Convert temperature from 55E liquid helium scale to ITS-90
    Raise ValueError if below T that corresponds to 0.65 K on ITS-90.
    Matches vapor pressure from 55E formulation to that of ITS-90.
    If T below 1.25 K on ITS-90, have to use T58 as intermediate.
    
    Units: 
    - T55E: K
    - Output: K (ITS-90)
    """
    
    Tlow = 0.64802 # corresponding to 0.65 K on ITS-90
    T90low = 1.248301 # corresponds to 1.25 K Lower limit of ITS-90 4He equation
    T5 = 4.988634 # corresponds to 5.0 K on ITS-90
    T522 = 5.21498 # corresponds to 5.22 K on T58 scale
      
    if T55E < Tlow or T55E > 5.22:
        raise ValueError(f"T55E must be between {Tlow} and 5.22 K.") 

    elif T55E < T90low:
    # Below ~1.25 K, since no ITS-90 vapor-pressure equation, have to use T58
    # as an intermediate. This produces a discontinuity of about 0.3 mK. 
        p55E = p_55E(T55E) 
        T58 = t_He58(p55E)
#        print(f"Calculated T58 from p55E: {T58:.6f} K")
        return t90_from_58(T58)
    
    elif T55E <= T5:
        p55E = p_55E(T55E)
        return t_He90(p55E) 
    
    elif T55E <= T522:
    # Above range of ITS-90 4He vapor pressure equation (but within T55E range),
    # use T58 as intermediate.
        p55E = p_55E(T55E) 
        T58 = t_He58(p55E)
        return t90_from_58(T58)
    
    else:
    # In the tiny range where T55E produces T58 above its upper limit of 5.22 K,
    # apply constant offset as reasonable approximation.
        return T55E + (t90_from_55E(T522) - T522)

def t90_from_L55(TL55):
    """
    Convert temperature from L55 liquid helium scale to ITS-90
    Raise ValueError if below 0.9 K lower limit of scale.
    Matches vapor pressure from L55 formulation to that of ITS-90.
    If T below 1.25 K on ITS-90, have to use T58 as intermediate.
    
    Units: 
    - TL55: K
    - Output: K (ITS-90)
    """
    
    Tlow = 0.9 # Lower limit of L55 scale
    T90low = 1.245064 # corresponds to 1.25 K Lower limit of ITS-90 4He equation
    T5 = 4.99429 # corresponds to 5.0 K on ITS-90
     
    if TL55 < Tlow or TL55 > 5.22:
        raise ValueError(f"TL55 must be between {Tlow} and 5.22 K.") 
    elif TL55 < T90low:
    # Below ~1.25 K, since no ITS-90 vapor-pressure equation, have to use T58
    # as an intermediate. This produces a discontinuity of about 3 mK. 
        pL55 = p_L55(TL55) 
        T58 = t_He58(pL55)
#        print(f"Calculated T58 from pL55: {T58:.6f} K")
        return t90_from_58(T58)
    
    elif TL55 <= T5:
        pL55 = p_L55(TL55)
        return t_He90(pL55) 
    
    else:
    # Above range of ITS-90 4He vapor pressure equation (but within T55E range),
    # use T58 as intermediate.
        pL55 = p_L55(TL55) 
        T58 = t_He58(pL55)
        return t90_from_58(T58)

def t90_from_He48(THe48):
    """
    Convert temperature from 1948 liquid helium scale to ITS-90
    Raise ValueError if below lower limit of 0.657 K.
    Matches vapor pressure from 1948 formulation to that of ITS-90.
    If T below 1.25 K on ITS-90, have to use T58 as intermediate.
    
    Units: 
    - THe48: K
    - Output: K (ITS-90)
    """
    
    Tlow = 0.657 # Lower limit of 1948 helium scale
    T90low = 1.247371 # corresponds to 1.25 K Lower limit of ITS-90 4He equation
    T5 = 5.01156 # corresponds to 5.0 K on ITS-90
     
    if THe48 < Tlow or THe48 > 5.2:
        raise ValueError(f"T on 1948 He scale must be between {Tlow} and 5.2 K.") 
    elif THe48 < T90low:
    # Below ~1.25 K, since no ITS-90 vapor-pressure equation, have to use T58
    # as an intermediate. This produces a discontinuity of about 3 mK. 
        pHe48 = p_He48(THe48) 
        T58 = t_He58(pHe48)
        return t90_from_58(T58)
    
    elif THe48 <= T5:
        pHe48 = p_He48(THe48)
        return t_He90(pHe48) 
    
    else:
    # Above range of ITS-90 4He vapor pressure equation (but within He48 range),
    # use T58 as intermediate.
        pHe48 = p_He48(THe48) 
        T58 = t_He58(pHe48)
        return t90_from_58(T58)

def t90_from_He37(THe37):
    """
    Convert temperature from 1937 liquid helium scale to ITS-90
    Matches vapor pressure from 1937 formulation of Keesom, 
    Physica 4, 971 (1937). Table IV gives differences T37 - T32.
    So convert to 1932 scale first and then to ITS-90.
    
    Units: 
    - THe37: K
    - Output: K (ITS-90)
    """
    
    Tlow = 1.0 # Lower limit
    THigh = 4.217 # Upper limit
     
    if THe37 < Tlow or THe37 > THigh:
        raise ValueError(f"T on 1937 He scale must be between {Tlow} and {THigh} K.") 
    # Cubic spline fit to Table IV differences, separately for above and below 2.19 K (on 32 scale)
    elif THe37 < 2.191:
    # Go from Fig. 1 to add a point at 1 K to slightly extend range
        T1_array = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.19])
        diff1_array = np.array([-2, 3, 10, 18, 23.5, 26, 26, 25, 22, 19, 16, 12, 9]) * 1e-3
        cs1 = CubicSpline(T1_array, diff1_array)
        diff1 = cs1(THe37)
        THe32 = THe37 - diff1
        return t90_from_He32(THe32)
    
    else:
    # Take point at 4.217 K from Table I to slighlty extend range to near He NBP
    # Need extra code to avoid unphysical spline wiggles at low end.
        if THe37 < 2.4:
            diff2 = 1e-3
        else:
            T2_array = np.array([2.19, 2.3, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.217])
            diff2_array = np.array([1, 1, 1, 2, 4, 7.5, 12, 14.5, 14.5, 11.5, 5.5, -3, -4]) * 1e-3
            cs2 = CubicSpline(T2_array, diff2_array)
            diff2 = cs2(THe37)
        THe32 = THe37 - diff2
        return t90_from_He32(THe32)

def t90_from_He32(THe32):
    """
    Convert temperature from 1932 liquid helium scale to ITS-90
    Matches vapor pressure from 1932 formulation of Keesom, 
    Leiden communication 219a (1932) to that of ITS-90.
    Equation on p. 142 of paper below 2.19 K, keeping 1929 equation above that.
    If T below 1.25 K or above 5 K on ITS-90, have to use T58 as intermediate.
    Note that this produces about a 8 mK discontinuity at 2.19 K.
    
    Units: 
    - THe32: K
    - Output: K (ITS-90)
    """
    
    Tlow = 0.65 # Lower limit
    T90low = 1.221255 # corresponds to 1.25 K Lower limit of ITS-90 4He equation
    T5 = 5.038135 # corresponds to 5.0 K on ITS-90
     
    if THe32 < Tlow or THe32 > 5.2:
        raise ValueError(f"T on 1932 He scale must be between {Tlow} and 5.2 K.") 
    elif THe32 < T90low:
    # Below ~1.25 K, since no ITS-90 vapor-pressure equation, have to use T58
    # as an intermediate. This produces a discontinuity of about 3 mK. 
    # Note that vapor pressure in cmHg
        lpcm = -3.018/THe32 + 2.484*np.log10(THe32) - 0.00297*THe32**4 + 1.197
        pHe32 = 10**lpcm * 101325. / 76
        T58 = t_He58(pHe32)
        return t90_from_58(T58)
    
    elif THe32 <= T5:
        if THe32 < 2.19:
            lpcm = -3.018/THe32 + 2.484*np.log10(THe32) - 0.00297*THe32**4 + 1.197
            pHe32 = 10**lpcm * 101325. / 76
            return t_He90(pHe32)
        else:
    # This is the same equation as in the 1929 He scale from Leiden
            lpcm = -3.024/THe32 + 2.208*np.log10(THe32) + 1.217
            pHe32 = 10**lpcm * 101325. / 76
            return t_He90(pHe32) 
    
    else:
    # Above range of ITS-90 4He vapor pressure equation (but within He29 range),
    # use T58 as intermediate.
        lpcm = -3.024/THe32 + 2.208*np.log10(THe32) + 1.217
        pHe32 = 10**lpcm * 101325. / 76
        T58 = t_He58(pHe32)
        return t90_from_58(T58)

def t90_from_He29(THe29):
    """
    Convert temperature from 1929 liquid helium scale to ITS-90
    Matches vapor pressure from formulation of Keesom et al., 
    Leiden communication 202c (1929) to that of ITS-90.
    Two equations (Eq. (6) in paper) for above and below 2.19 K
    If T below 1.25 K or above 5 K on ITS-90, have to use T58 as intermediate.
    
    Units: 
    - THe29: K
    - Output: K (ITS-90)
    """
    
    Tlow = 0.65 # Lower limit
    T90low = 1.214302 # corresponds to 1.25 K Lower limit of ITS-90 4He equation
    T5 = 5.038135 # corresponds to 5.0 K on ITS-90
     
    if THe29 < Tlow or THe29 > 5.2:
        raise ValueError(f"T on 1929 He scale must be between {Tlow} and 5.2 K.") 
    elif THe29 < T90low:
    # Below ~1.25 K, since no ITS-90 vapor-pressure equation, have to use T58
    # as an intermediate. This produces a discontinuity of about 3 mK. 
    # Note that 1929 paper vapor pressure in cmHg
        lpcm = -3.859/THe29 + 0.922*np.log10(THe29) + 2.035
        pHe29 = 10**lpcm * 101325. / 76
        T58 = t_He58(pHe29)
        return t90_from_58(T58)
    
    elif THe29 <= T5:
        if THe29 < 2.19:
            lpcm = -3.859/THe29 + 0.922*np.log10(THe29) + 2.035
            pHe29 = 10**lpcm * 101325. / 76
            return t_He90(pHe29)
        else:
            lpcm = -3.024/THe29 + 2.208*np.log10(THe29) + 1.217
            pHe29 = 10**lpcm * 101325. / 76
            return t_He90(pHe29) 
    
    else:
    # Above range of ITS-90 4He vapor pressure equation (but within He29 range),
    # use T58 as intermediate.
        lpcm = -3.024/THe29 + 2.208*np.log10(THe29) + 1.217
        pHe29 = 10**lpcm * 101325. / 76
        T58 = t_He58(pHe29)
        return t90_from_58(T58)

def t90_from_He24(THe24):
    """
    Convert temperature from 1924 liquid helium scale to ITS-90
    Matches vapor pressure from 1924 formulation to that of ITS-90.
    Vapor pressure from Verschaffelt (Leiden Supplement 49, 1924) is used below 1.5 K,
    and from Kamerlingh Onnes & Weber (1924) above that.
    If T below 1.25 K or above 5 K on ITS-90, have to use T58 as intermediate.
    
    Units: 
    - THe24: K
    - Output: K (ITS-90)
    """
    
    Tlow = 0.65 # Lower limit
    T90low = 1.246132 # corresponds to 1.25 K Lower limit of ITS-90 4He equation
    T5 = 5.015137 # corresponds to 5.0 K on ITS-90
     
    if THe24 < Tlow or THe24 > 5.2:
        raise ValueError(f"T on 1924 He scale must be between {Tlow} and 5.2 K.") 
    elif THe24 < T90low:
    # Below ~1.25 K, since no ITS-90 vapor-pressure equation, have to use T58
    # as an intermediate. This produces a discontinuity of about 3 mK. 
    # Note that 1924 Verschaffelt equation gives vapor pressure in atmospheres
        lpatm = -3.1/THe24 + 2.5*np.log10(THe24) - 0.69
        pHe24 = 10**lpatm * 101325. 
        T58 = t_He58(pHe24)
        return t90_from_58(T58)
    
    elif THe24 <= T5:
        if THe24 < 1.5:
            lpatm = -3.1/THe24 + 2.5*np.log10(THe24) - 0.69
            pHe24 = 10**lpatm * 101325. 
            return t_He90(pHe24)
        else:
    # 1915 Kamerlingh Onnes & Weber equation (p. 507) gives vapor pressure in cm Hg
            lpcm = 3.729 - 7.978/THe24 - 0.13628/np.square(THe24) + 4.3634 / THe24**3
            pHe24 = 10**lpcm * 101325. / 76
            return t_He90(pHe24) 
    
    else:
    # Above range of ITS-90 4He vapor pressure equation (but within He24 range),
    # use T58 as intermediate.
        lpcm = 3.729 - 7.978/THe24 - 0.13628/np.square(THe24) + 4.3634 / THe24**3
        pHe24 = 10**lpcm * 101325. / 76
        T58 = t_He58(pHe24)
        return t90_from_58(T58)

def t90_from_HeBS(THeBS):
    """
    Convert temperature from 1939 liquid helium scale of Bleaney and Simon to ITS-90.
    Matches vapor pressure, only below lambda point which is said to be 2.186 K.
    Bleaney and Simon, Trans. Faraday Soc. (1939), p.1205.
    If T below 1.25 K on ITS-90, have to use T58 as intermediate.
    
    Units: 
    - THeBS: K
    - Output: K (ITS-90)
    """
    
    Tlow = 0.64742 # Lower limit
    T90low = 1.247289 # corresponds to 1.25 K Lower limit of ITS-90 4He equation
    Tlam = 2.186 # Lambda point, upper limit of scale
     
    if THeBS < Tlow or THeBS > Tlam:
        raise ValueError(f"T on 1939 Bleaney-Simon He scale must be between {Tlow} and {Tlam} K.") 
    else:
    # Implement their vapor pressure equation, which requires a Delta term read out
    # from their Fig. 4. 
        T_array = np.array([
        0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 
        1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.186
        ])
        Delta_array = np.array([
        0.0000, 0.0001, 0.0003, 0.0006, 0.0009, 0.0014, 0.0020, 0.0026, 0.0030, 0.0032,
        0.0029, 0.0023, 0.0012, 0.0000, -0.0019, -0.0050, -0.0091
        ])
   
        cs = CubicSpline(T_array, Delta_array)
        Delta = cs(THeBS)
        lpcm = -3.117/THeBS + 2.5*np.log10(THeBS) - 0.00227*THeBS**3 + 1.196 + Delta
        pHeBS = 10**lpcm * 101325. / 76
        if THeBS < T90low:
    # Below ~1.25 K, since no ITS-90 vapor-pressure equation, have to use T58
    # as an intermediate. This produces a discontinuity of about 3 mK. 
            T58 = t_He58(pHeBS)
            return t90_from_58(T58)
        else:
            return t_He90(pHeBS)

def p_He48(THe48):
    """
    Calculates the vapor pressure of 4He (in mm Hg at 0°C) using 
    linear interpolation on Table I from Annexe T15 of Vol. B28 of
    Proc. Verb. CIPM (1952)
    Uses formula in this paper above 4.3 K; this creates a discontinuity 
    of about 96 Pa at 4.3 K.

    Units: 
    - TL55: K
    - Output: Pa
    """
    # Boundary check based on the limits of Table V 
    if THe48 < 0.657 or THe48 > 5.20:
        raise ValueError("Temperature on 1948 helium scale must be between 0.657 K and 5.20 K.")

    if THe48 >= 4.3:
    # Use equation on page T 152 of paper for T above 4.3 K.
        a = -3.52e-7
        b = 1.905e-3
        c_base = 2.967
    
    # Quadratic equation: a*P^2 + b*P + (c_base - T) = 0
        c = c_base - THe48
    
    # Calculate discriminant: b^2 - 4ac
        discriminant = b**2 - 4 * a * c
    
    # Quadratic formula: (-b +/- sqrt(discriminant)) / (2a)
    # We use the subtractive version of -b because 'a' is negative, 
    # ensuring a positive pressure result.
        p_mmHg20 = (-b + np.sqrt(discriminant)) / (2 * a)
        p_mmHg = p_mmHg20 * 0.99638
        return p_mmHg / 760.0 * 101325.0
    
    # Otherwise interpolate Pressure values from Table I 
    T_1 = np.array([
    # p = 0.0010 to 0.0099 (Step: 0.0001)
    0.657, 0.662, 0.667, 0.671, 0.674, 0.678, 0.681, 0.684, 0.687, 0.690,
    0.693, 0.696, 0.698, 0.701, 0.703, 0.706, 0.708, 0.710, 0.712, 0.714,
    0.716, 0.717, 0.719, 0.721, 0.723, 0.724, 0.726, 0.728, 0.729, 0.731,
    0.732, 0.734, 0.735, 0.737, 0.738, 0.740, 0.741, 0.742, 0.744, 0.745,
    0.746, 0.747, 0.749, 0.750, 0.751, 0.752, 0.753, 0.754, 0.755, 0.756,
    0.757, 0.758, 0.759, 0.760, 0.761, 0.762, 0.763, 0.764, 0.765, 0.766,
    0.767, 0.768, 0.769, 0.770, 0.771, 0.772, 0.773, 0.773, 0.774, 0.775,
    0.776, 0.777, 0.778, 0.778, 0.779, 0.780, 0.781, 0.781, 0.782, 0.783,
    0.784, 0.785, 0.785, 0.786, 0.787, 0.787, 0.788, 0.789, 0.790, 0.790,
    0.791
    ])
    # p = 0.010 to 0.099 (Step: 0.001)
    T_2 = np.array([
    0.791, 0.797, 0.803, 0.809, 0.815, 0.819, 0.824, 0.829, 0.833, 0.837,
    0.841, 0.845, 0.848, 0.852, 0.855, 0.858, 0.862, 0.865, 0.867, 0.870,
    0.873, 0.875, 0.878, 0.881, 0.883, 0.886, 0.888, 0.890, 0.893, 0.895,
    0.897, 0.899, 0.901, 0.903, 0.905, 0.907, 0.909, 0.911, 0.913, 0.915,
    0.916, 0.918, 0.920, 0.922, 0.923, 0.925, 0.927, 0.928, 0.930, 0.931,
    0.933, 0.934, 0.936, 0.937, 0.939, 0.940, 0.941, 0.943, 0.944, 0.946,
    0.947, 0.948, 0.950, 0.951, 0.952, 0.953, 0.955, 0.956, 0.957, 0.958,
    0.960, 0.961, 0.962, 0.963, 0.964, 0.965, 0.967, 0.968, 0.969, 0.970,
    0.971, 0.972, 0.973, 0.974, 0.975, 0.976, 0.977, 0.979, 0.980, 0.981,
    0.982
    ])
    # p = 0.10 to 0.99 (Step: 0.01) - Rollover to 1.0K included
    T_3 = np.array([
    0.982, 0.991, 1.000, 1.009, 1.016, 1.024, 1.030, 1.036, 1.042, 1.048,
    1.054, 1.060, 1.065, 1.070, 1.076, 1.080, 1.085, 1.090, 1.094, 1.099,
    1.103, 1.107, 1.111, 1.115, 1.118, 1.122, 1.125, 1.129, 1.132, 1.136,
    1.139, 1.142, 1.145, 1.149, 1.152, 1.155, 1.158, 1.161, 1.164, 1.166,
    1.169, 1.172, 1.175, 1.177, 1.180, 1.182, 1.185, 1.187, 1.189, 1.192,
    1.194, 1.196, 1.199, 1.201, 1.203, 1.205, 1.208, 1.210, 1.212, 1.214,
    1.216, 1.218, 1.220, 1.222, 1.224, 1.226, 1.228, 1.230, 1.232, 1.234,
    1.236, 1.238, 1.239, 1.241, 1.243, 1.245, 1.247, 1.248, 1.250, 1.252,
    1.253, 1.255, 1.257, 1.258, 1.260, 1.261, 1.263, 1.265, 1.266, 1.268,
    1.269
    ])
    # p = 1.0 to 39.9 (Step: 0.1)
    T_4 = np.array([
    1.269, 1.281, 1.298, 1.312, 1.324, 1.336, 1.347, 1.358, 1.368, 1.377,
    1.387, 1.396, 1.404, 1.412, 1.420, 1.428, 1.435, 1.443, 1.450, 1.457,
    1.463, 1.470, 1.476, 1.482, 1.489, 1.495, 1.500, 1.506, 1.512, 1.517,
    1.523, 1.528, 1.534, 1.539, 1.544, 1.549, 1.554, 1.559, 1.563, 1.568,
    1.572, 1.577, 1.581, 1.586, 1.590, 1.594, 1.598, 1.603, 1.607, 1.611,
    1.615, 1.618, 1.622, 1.626, 1.630, 1.634, 1.637, 1.641, 1.644, 1.648,
    1.651, 1.655, 1.658, 1.662, 1.665, 1.668, 1.672, 1.675, 1.678, 1.682,
    1.685, 1.688, 1.691, 1.694, 1.697, 1.700, 1.703, 1.706, 1.709, 1.712,
    1.715, 1.718, 1.721, 1.724, 1.726, 1.729, 1.732, 1.735, 1.737, 1.740,
    1.743, 1.745, 1.748, 1.750, 1.753, 1.755, 1.758, 1.761, 1.763, 1.766,
    1.768, 1.771, 1.773, 1.775, 1.778, 1.780, 1.783, 1.785, 1.787, 1.790,
    1.792, 1.794, 1.797, 1.799, 1.801, 1.804, 1.806, 1.808, 1.810, 1.812,
    1.815, 1.817, 1.819, 1.821, 1.824, 1.826, 1.828, 1.830, 1.832, 1.834,
    1.836, 1.839, 1.841, 1.843, 1.845, 1.847, 1.849, 1.851, 1.853, 1.855,
    1.857, 1.859, 1.861, 1.863, 1.865, 1.867, 1.869, 1.871, 1.873, 1.875,
    1.877, 1.879, 1.881, 1.883, 1.884, 1.886, 1.888, 1.890, 1.892, 1.894,
    1.895, 1.897, 1.899, 1.901, 1.903, 1.904, 1.906, 1.908, 1.910, 1.912,
    1.913, 1.915, 1.917, 1.918, 1.920, 1.922, 1.924, 1.925, 1.927, 1.929,
    1.930, 1.932, 1.934, 1.935, 1.937, 1.939, 1.940, 1.942, 1.944, 1.945,
    1.947, 1.948, 1.950, 1.952, 1.953, 1.955, 1.956, 1.958, 1.960, 1.961,
    1.963, 1.964, 1.966, 1.967, 1.969, 1.970, 1.972, 1.974, 1.975, 1.977,
    1.978, 1.980, 1.981, 1.983, 1.984, 1.986, 1.987, 1.989, 1.990, 1.992,
    1.993, 1.995, 1.996, 1.998, 1.999, 2.001, 2.002, 2.004, 2.005, 2.007,
    2.008, 2.010, 2.011, 2.012, 2.014, 2.015, 2.017, 2.018, 2.020, 2.021,
    2.023, 2.024, 2.025, 2.027, 2.028, 2.030, 2.031, 2.033, 2.034, 2.035,
    2.037, 2.038, 2.039, 2.041, 2.042, 2.044, 2.045, 2.046, 2.048, 2.049,
    2.050, 2.052, 2.053, 2.054, 2.056, 2.057, 2.058, 2.060, 2.061, 2.062,
    2.064, 2.065, 2.066, 2.068, 2.069, 2.070, 2.072, 2.073, 2.074, 2.076,
    2.077, 2.078, 2.079, 2.081, 2.082, 2.083, 2.085, 2.086, 2.087, 2.088,
    2.090, 2.091, 2.092, 2.093, 2.095, 2.096, 2.097, 2.098, 2.100, 2.101,
    2.102, 2.103, 2.105, 2.106, 2.107, 2.108, 2.109, 2.111, 2.112, 2.113,
    2.114, 2.115, 2.117, 2.118, 2.119, 2.120, 2.121, 2.123, 2.124, 2.125,
    2.126, 2.127, 2.128, 2.130, 2.131, 2.132, 2.133, 2.134, 2.135, 2.137,
    2.138, 2.139, 2.140, 2.141, 2.142, 2.144, 2.145, 2.146, 2.147, 2.148,
    2.149, 2.150, 2.151, 2.153, 2.154, 2.155, 2.156, 2.157, 2.158, 2.159,
    2.160, 2.162, 2.163, 2.164, 2.165, 2.166, 2.167, 2.168, 2.169, 2.170,
    2.171, 2.172, 2.174, 2.175, 2.176, 2.177, 2.178, 2.179, 2.180, 2.181,
    2.182, 2.183, 2.184, 2.185, 2.186, 2.187, 2.188, 2.190, 2.191, 2.192,
    2.193, 2.194, 2.195, 2.196, 2.197, 2.198, 2.199, 2.200, 2.201, 2.202,
    2.203
    ])
    # p = 40 to 799 (Step: 1)
    T_5 = np.array([
    2.203, 2.213, 2.223, 2.233, 2.243, 2.253, 2.262, 2.271, 2.280, 2.289,
    2.298, 2.307, 2.315, 2.324, 2.332, 2.340, 2.348, 2.356, 2.364, 2.372,
    2.380, 2.388, 2.395, 2.403, 2.410, 2.418, 2.425, 2.432, 2.439, 2.446,
    2.453, 2.460, 2.467, 2.474, 2.481, 2.487, 2.494, 2.501, 2.507, 2.513,
    2.520, 2.526, 2.533, 2.539, 2.545, 2.551, 2.557, 2.563, 2.569, 2.575,
    2.581, 2.587, 2.593, 2.599, 2.604, 2.610, 2.616, 2.621, 2.627, 2.633,
    2.638, 2.644, 2.649, 2.655, 2.660, 2.665, 2.671, 2.676, 2.681, 2.686,
    2.692, 2.697, 2.701, 2.706, 2.711, 2.717, 2.722, 2.726, 2.731, 2.736,
    2.741, 2.746, 2.750, 2.755, 2.760, 2.765, 2.770, 2.774, 2.779, 2.783,
    2.788, 2.792, 2.797, 2.801, 2.806, 2.810, 2.815, 2.820, 2.824, 2.829,
    2.833, 2.837, 2.842, 2.846, 2.850, 2.855, 2.859, 2.863, 2.868, 2.872,
    2.877, 2.881, 2.885, 2.889, 2.893, 2.897, 2.902, 2.906, 2.910, 2.914,
    2.918, 2.922, 2.926, 2.930, 2.934, 2.938, 2.942, 2.946, 2.950, 2.954,
    2.958, 2.962, 2.965, 2.969, 2.973, 2.977, 2.981, 2.984, 2.988, 2.992,
    2.996, 2.999, 3.003, 3.007, 3.010, 3.014, 3.018, 3.021, 3.025, 3.028,
    3.032, 3.035, 3.039, 3.042, 3.046, 3.050, 3.053, 3.057, 3.060, 3.063,
    3.067, 3.070, 3.074, 3.077, 3.081, 3.084, 3.087, 3.091, 3.094, 3.098,
    3.101, 3.104, 3.108, 3.111, 3.114, 3.118, 3.121, 3.124, 3.128, 3.131,
    3.134, 3.138, 3.141, 3.144, 3.147, 3.151, 3.154, 3.157, 3.160, 3.164,
    3.167, 3.170, 3.173, 3.176, 3.180, 3.183, 3.186, 3.189, 3.192, 3.195,
    3.198, 3.202, 3.205, 3.208, 3.211, 3.214, 3.217, 3.220, 3.223, 3.226,
    3.229, 3.232, 3.235, 3.238, 3.241, 3.244, 3.247, 3.250, 3.253, 3.255,
    3.258, 3.261, 3.264, 3.267, 3.270, 3.273, 3.275, 3.278, 3.281, 3.284,
    3.287, 3.289, 3.292, 3.295, 3.298, 3.301, 3.303, 3.306, 3.309, 3.312,
    3.314, 3.317, 3.320, 3.322, 3.325, 3.328, 3.331, 3.333, 3.336, 3.339,
    3.341, 3.344, 3.347, 3.349, 3.352, 3.355, 3.357, 3.360, 3.363, 3.365,
    3.368, 3.371, 3.373, 3.376, 3.378, 3.381, 3.384, 3.386, 3.389, 3.391,
    3.394, 3.396, 3.399, 3.402, 3.404, 3.407, 3.409, 3.412, 3.414, 3.417,
    3.419, 3.422, 3.424, 3.427, 3.429, 3.432, 3.434, 3.437, 3.439, 3.442,
    3.445, 3.447, 3.449, 3.452, 3.454, 3.456, 3.459, 3.461, 3.464, 3.466,
    3.468, 3.471, 3.473, 3.475, 3.478, 3.480, 3.483, 3.485, 3.487, 3.490,
    3.492, 3.494, 3.497, 3.499, 3.501, 3.504, 3.506, 3.508, 3.511, 3.513,
    3.515, 3.518, 3.520, 3.522, 3.525, 3.527, 3.529, 3.531, 3.534, 3.536,
    3.538, 3.540, 3.543, 3.545, 3.547, 3.550, 3.552, 3.554, 3.556, 3.558,
    3.561, 3.563, 3.565, 3.567, 3.570, 3.572, 3.574, 3.576, 3.578, 3.581,
    3.583, 3.585, 3.587, 3.589, 3.592, 3.594, 3.596, 3.598, 3.600, 3.602,
    3.605, 3.607, 3.609, 3.611, 3.613, 3.615, 3.617, 3.620, 3.622, 3.624,
    3.626, 3.628, 3.630, 3.632, 3.634, 3.637, 3.639, 3.641, 3.643, 3.645,
    3.647, 3.649, 3.651, 3.653, 3.655, 3.658, 3.660, 3.662, 3.664, 3.666,
    3.668, 3.670, 3.672, 3.674, 3.676, 3.678, 3.680, 3.682, 3.684, 3.686,
    3.688, 3.690, 3.692, 3.694, 3.696, 3.698, 3.700, 3.702, 3.704, 3.706,
    3.708, 3.710, 3.712, 3.714, 3.716, 3.718, 3.720, 3.722, 3.724, 3.726,
    3.727, 3.729, 3.731, 3.733, 3.735, 3.737, 3.739, 3.741, 3.743, 3.745,
    3.747, 3.749, 3.751, 3.752, 3.754, 3.756, 3.758, 3.760, 3.762, 3.764,
    3.766, 3.768, 3.769, 3.771, 3.773, 3.775, 3.777, 3.779, 3.781, 3.783,
    3.784, 3.786, 3.788, 3.790, 3.792, 3.794, 3.796, 3.797, 3.799, 3.801,
    3.803, 3.805, 3.807, 3.809, 3.810, 3.812, 3.814, 3.816, 3.818, 3.819,
    3.821, 3.823, 3.825, 3.827, 3.829, 3.830, 3.832, 3.834, 3.836, 3.838,
    3.839, 3.841, 3.843, 3.845, 3.846, 3.848, 3.850, 3.852, 3.853, 3.855,
    3.857, 3.859, 3.861, 3.862, 3.864, 3.866, 3.868, 3.869, 3.871, 3.873,
    3.875, 3.876, 3.878, 3.880, 3.882, 3.883, 3.885, 3.887, 3.888, 3.890,
    3.892, 3.894, 3.895, 3.897, 3.899, 3.900, 3.902, 3.904, 3.906, 3.907,
    3.909, 3.911, 3.912, 3.914, 3.916, 3.917, 3.919, 3.921, 3.922, 3.924,
    3.926, 3.927, 3.929, 3.931, 3.932, 3.934, 3.936, 3.937, 3.939, 3.941,
    3.942, 3.944, 3.946, 3.947, 3.949, 3.951, 3.952, 3.954, 3.956, 3.957,
    3.959, 3.961, 3.962, 3.964, 3.965, 3.967, 3.969, 3.970, 3.972, 3.974,
    3.975, 3.977, 3.978, 3.980, 3.982, 3.983, 3.985, 3.986, 3.988, 3.990,
    3.991, 3.993, 3.994, 3.996, 3.997, 3.999, 4.001, 4.002, 4.004, 4.005,
    4.007, 4.008, 4.010, 4.012, 4.013, 4.015, 4.016, 4.018, 4.019, 4.021,
    4.022, 4.024, 4.026, 4.027, 4.029, 4.030, 4.032, 4.033, 4.035, 4.036,
    4.038, 4.039, 4.041, 4.042, 4.044, 4.045, 4.047, 4.048, 4.050, 4.051,
    4.053, 4.055, 4.056, 4.058, 4.059, 4.061, 4.062, 4.064, 4.065, 4.067,
    4.068, 4.070, 4.071, 4.073, 4.074, 4.076, 4.077, 4.079, 4.080, 4.082,
    4.083, 4.084, 4.086, 4.087, 4.089, 4.090, 4.092, 4.093, 4.095, 4.096,
    4.098, 4.099, 4.101, 4.102, 4.104, 4.105, 4.107, 4.108, 4.110, 4.111,
    4.112, 4.114, 4.115, 4.117, 4.118, 4.120, 4.121, 4.123, 4.124, 4.126,
    4.127, 4.128, 4.130, 4.131, 4.133, 4.134, 4.136, 4.137, 4.138, 4.140,
    4.141, 4.143, 4.144, 4.146, 4.147, 4.148, 4.150, 4.151, 4.153, 4.154,
    4.156, 4.157, 4.158, 4.160, 4.161, 4.163, 4.164, 4.165, 4.167, 4.168,
    4.170, 4.171, 4.173, 4.174, 4.175, 4.177, 4.178, 4.180, 4.181, 4.182,
    4.184, 4.185, 4.187, 4.188, 4.189, 4.191, 4.192, 4.193, 4.195, 4.196,
    4.198, 4.199, 4.200, 4.202, 4.203, 4.205, 4.206, 4.207, 4.209, 4.210,
    4.211, 4.213, 4.214, 4.216, 4.217, 4.218, 4.220, 4.221, 4.222, 4.224,
    4.225, 4.226, 4.228, 4.229, 4.231, 4.232, 4.233, 4.235, 4.236, 4.237,
    4.239, 4.240, 4.241, 4.243, 4.244, 4.245, 4.247, 4.248, 4.250, 4.251,
    4.252, 4.254, 4.255, 4.256, 4.258, 4.259, 4.260, 4.262, 4.263, 4.264,
    4.266, 4.267, 4.268, 4.270, 4.271, 4.272, 4.274, 4.275, 4.276, 4.278,
    4.28
    ])
    # p = 810 to 1720 (Step: 10)
    T_6 = np.array([
          4.28, 4.29, 4.31, 4.32, 4.33, 4.34, 4.36, 4.37, 4.38,
    4.40, 4.41, 4.42, 4.43, 4.45, 4.46, 4.47, 4.48, 4.50, 4.51,
    4.52, 4.53, 4.54, 4.56, 4.57, 4.58, 4.59, 4.60, 4.61, 4.63,
    4.64, 4.65, 4.66, 4.67, 4.68, 4.69, 4.70, 4.71, 4.72, 4.74,
    4.75, 4.76, 4.77, 4.78, 4.79, 4.80, 4.81, 4.82, 4.83, 4.84,
    4.85, 4.86, 4.87, 4.88, 4.89, 4.90, 4.91, 4.92, 4.93, 4.93,
    4.94, 4.95, 4.96, 4.97, 4.98, 4.99, 5.00, 5.01, 5.02, 5.02,
    5.03, 5.04, 5.05, 5.06, 5.07, 5.07, 5.08, 5.09, 5.10, 5.11,
    5.11, 5.12, 5.13, 5.14, 5.14, 5.15, 5.16, 5.17, 5.17, 5.18,
    5.19, 5.20, 5.20
    ])

# Corresponding Pressure (p) Array (mm Hg at 20°C)
#    p_values = np.concatenate([
    p_1 = np.linspace(0.0010, 0.0100, 91)   # Seg 1: Step 0.0001
    p_2 = np.linspace(0.010, 0.10, 91)    # Seg 2: Step 0.001
    p_3 = np.linspace(0.10, 1.00, 91)      # Seg 3: Step 0.01
    p_4 = np.linspace(1.0, 40, 391)      # Seg 4: Step 0.1
    p_5 = np.linspace(40, 810, 771)        # Seg 5: Step 1
    p_6 = np.linspace(810, 1720, 92)        # Seg 6: Step 10

    # Perform linear interpolation based on which segment of the table
    # the temperature falls into.
    if THe48 <= 0.791:
        p_mmHg20 = np.interp(THe48, T_1, p_1)
    elif THe48 <= 0.982:
        p_mmHg20 = np.interp(THe48, T_2, p_2)
    elif THe48 <= 1.269:
        p_mmHg20 = np.interp(THe48, T_3, p_3)
    elif THe48 <= 2.203:
        p_mmHg20 = np.interp(THe48, T_4, p_4)
    elif THe48 <= 4.28:
        p_mmHg20 = np.interp(THe48, T_5, p_5)
    else:
        p_mmHg20 = np.interp(THe48, T_6, p_6)
# Convert mmHg to from 20 degC to 0 degC, then to Pascals
    p_mmHg = p_mmHg20 * 0.99638
    return p_mmHg / 760.0 * 101325.0

def p_55E(T55E):
    """
    Calculates the vapor pressure of 4He (in mm Hg at 0°C) using 
    Eqs. (1) and (2) in J.R. Clement, Proc. Verb. CIPM, 26A, Annex T21 (1958)
    Different equations above and below lambda point, which is taken as 2.1735 K.

    Units: 
    - T55E: K
    - Output: Pa
    """
    # Boundary check based on stated limits 
    if T55E < 0.5 or T55E > 5.22:
        raise ValueError("Temperature TL55 must be between 0.5 K and 5.22 K.")
    if T55E >= 2.1735:
        lnp = 6.22077 - 8.3861/T55E + 0.945*np.log(T55E) + 0.2475*T55E
    else:
        lnp = 5.04862 - 7.18132/T55E + 2.5*np.log(T55E) - 4.75e-5*T55E**9
    p_mmHg = np.exp(lnp)
# Convert mmHg to Pascals
    return p_mmHg /760.0 * 101325.0

@lru_cache(maxsize=1)
def get_HeL55_spline():
    """
    Defines the data and fits the spline only once for L55 He vapor pressure
    Uses Table V from H. van Dijk and M. Durieux,
    "Thermodynamic Temperature Scale (T_L55) in the Liquid Helium Region,
    Physica 24, 1-19 (1958).
    """
    # Pressure values from Table V 
    p_values = np.array([
        # 0.9 - 1.8 K
        0.04224, 0.04738, 0.05304, 0.05924, 0.06604, 0.07346, 0.08157, 0.09039, 0.09999, 0.11041,
        0.12170, 0.13393, 0.14714, 0.16140, 0.17676, 0.19330, 0.21108, 0.23016, 0.25062, 0.27253,
        0.29597, 0.32101, 0.34774, 0.37623, 0.40657, 0.43885, 0.47316, 0.50958, 0.54822, 0.58917,
        0.63253, 0.67840, 0.72688, 0.77808, 0.83211, 0.88907, 0.94909, 1.01226, 1.07872, 1.14858,
        1.22196, 1.29899, 1.37979, 1.46448, 1.55321, 1.64609, 1.74326, 1.84487, 1.95103, 2.06190,
        2.17762, 2.29833, 2.42416, 2.55528, 2.69183, 2.83395, 2.98180, 3.13553, 3.29529, 3.46124,
        3.63354, 3.81234, 3.99780, 4.19008, 4.38934, 4.59575, 4.80946, 5.03064, 5.25946, 5.49607,
        5.74065, 5.99336, 6.25438, 6.52385, 6.80196, 7.08887, 7.38474, 7.68975, 8.00406, 8.32783,
        8.66124, 9.00445, 9.35762, 9.72093, 10.0945, 10.4786, 10.8733, 11.2788, 11.6952, 12.1227,
        12.5614, 13.0116, 13.4733, 13.9467, 14.4320, 14.9293, 15.4388, 15.9606, 16.4948, 17.0415,
        # 1.9 - 2.8 K
        17.6010, 18.1734, 18.7587, 19.3570, 19.9685, 20.5933, 21.2316, 21.8833, 22.5485, 23.2274,
        23.9199, 24.6264, 25.3467, 26.0809, 26.8289, 27.5909, 28.3670, 29.1571, 29.9610, 30.7789,
        31.6106, 32.4565, 33.3161, 34.1894, 35.0762, 35.9764, 36.8896, 37.8152, 38.7536, 39.7069,
        40.6754, 41.6597, 42.6597, 43.6757, 44.7078, 45.7564, 46.8217, 47.9038, 49.0030, 50.1193,
        51.2531, 52.4041, 53.5728, 54.7591, 55.9634, 57.1858, 58.4264, 59.6854, 60.9628, 62.2590,
        63.5739, 64.9077, 66.2607, 67.6328, 69.0244, 70.4354, 71.8661, 73.3166, 74.7871, 76.2776,
        77.7884, 79.3196, 80.8713, 82.4436, 84.0368, 85.6509, 87.2861, 88.9425, 90.6203, 92.3197,
        94.0407, 95.7837, 97.5486, 99.3356, 101.145, 102.977, 104.831, 106.707, 108.607, 110.529,
        112.474, 114.443, 116.435, 118.450, 120.489, 122.552, 124.639, 126.749, 128.884, 131.043,
        133.227, 135.435, 137.668, 139.926, 142.208, 144.516, 146.849, 149.208, 151.592, 154.001,
        # 2.9 - 3.8 K
        156.437, 158.899, 161.386, 163.900, 166.440, 169.007, 171.600, 174.220, 176.867, 179.541,
        182.242, 184.971, 187.727, 190.511, 193.322, 196.162, 199.029, 201.925, 204.849, 207.801,
        210.782, 213.792, 216.830, 219.898, 222.994, 226.120, 229.275, 232.460, 235.674, 238.918,
        242.192, 245.496, 248.830, 252.195, 255.590, 259.016, 262.473, 265.960, 269.479, 273.029,
        276.610, 280.222, 283.866, 287.542, 291.249, 294.989, 298.761, 302.565, 306.402, 310.272,
        314.174, 318.109, 322.077, 326.078, 330.113, 334.181, 338.283, 342.418, 346.588, 350.792,
        355.030, 359.303, 363.610, 367.952, 372.329, 376.741, 381.188, 385.671, 390.189, 394.744,
        399.334, 403.960, 408.623, 413.323, 418.059, 422.832, 427.642, 432.490, 437.375, 442.297,
        447.258, 452.257, 457.293, 462.368, 467.482, 472.635, 477.827, 483.058, 488.328, 493.638,
        498.988, 504.378, 509.808, 515.279, 520.790, 526.342, 531.935, 537.570, 543.246, 548.963,
        # 3.9 - 4.8 K
        554.723, 560.524, 566.368, 572.254, 578.183, 584.156, 590.172, 596.233, 602.337, 608.486,
        614.680, 620.919, 627.203, 633.533, 639.909, 646.332, 652.801, 659.317, 665.881, 672.492,
        679.152, 685.860, 692.617, 699.424, 706.281, 713.188, 720.146, 727.156, 734.212, 741.313,
        748.459, 755.649, 762.886, 770.167, 777.494, 784.867, 792.285, 799.749, 807.259, 814.816,
        822.418, 830.067, 837.762, 845.504, 853.293, 861.129, 869.011, 876.941, 884.918, 892.942,
        901.013, 909.132, 917.299, 925.514, 933.777, 942.087, 950.446, 958.853, 967.309, 975.813,
        984.366, 992.967, 1001.62, 1010.32, 1019.07, 1027.86, 1036.71, 1045.61, 1054.55, 1063.55,
        1072.60, 1081.69, 1090.84, 1100.03, 1109.28, 1118.57, 1127.92, 1137.32, 1146.77, 1156.27,
        1165.82, 1175.42, 1185.07, 1194.78, 1204.53, 1214.34, 1224.20, 1234.11, 1244.07, 1254.08,
        1264.15, 1274.27, 1284.44, 1294.66, 1304.94, 1315.27, 1325.65, 1336.09, 1346.57, 1357.11,
        # 4.9 - 5.22 K
        1367.71, 1378.36, 1389.06, 1399.81, 1410.62, 1421.48, 1432.40, 1443.37, 1454.39, 1465.47,
        1476.60, 1487.79, 1499.03, 1510.33, 1521.68, 1533.08, 1544.54, 1556.06, 1567.63, 1579.26,
        1590.94, 1602.68, 1614.47, 1626.32, 1638.23, 1650.19, 1662.21, 1674.28, 1686.41, 1698.59,
        1710.84, 1723.14, 1735.49
    ])

    # Generate temperature grid at 0.01 K intervals 
    t_points = np.arange(0.9, 5.225, 0.01)

    return CubicSpline(t_points, p_values)

def p_L55(TL55):
    """
    Calculates the vapor pressure of 4He (in mm Hg at 0°C) using 
    cubic spline interpolation on Table V from H. van Dijk and M. Durieux,
    "Thermodynamic Temperature Scale (T_L55) in the Liquid Helium Region,
    Physica 24, 1-19 (1958).

    Units: 
    - TL55: K
    - Output: Pa
    """
    # Boundary check based on the limits of Table V 
    if TL55 < 0.9 or TL55 > 5.22:
        raise ValueError("Temperature TL55 must be between 0.9 K and 5.22 K.")

    # Perform cubic spline interpolation 
    spline = get_HeL55_spline()
    p_mmHg = float(spline(TL55))
# Convert mmHg to Pascals
    return p_mmHg /760.0 * 101325.0

@lru_cache(maxsize=1)
def get_He58_spline():
    """
    Defines the data and fits the spline only once for 1958 He vapor pressure
    Taken from Table I of H. Van Dijk et al.,
    "Vapour Pressure Temperature Scale for the Liquid 4He Region," Physica 24, S129-S131 (1958)
    """
    # Temperature T in Kelvin (0.50 to 5.22)
    t_data = np.array([
        0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
        0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
        0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
        0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
        0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09,
        1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19,
        1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29,
        1.30, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39,
        1.40, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49,
        1.50, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59,
        1.60, 1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69,
        1.70, 1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79,
        1.80, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89,
        1.90, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99,
        2.00, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08, 2.09,
        2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19,
        2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28, 2.29,
        2.30, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39,
        2.40, 2.41, 2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48, 2.49,
        2.50, 2.51, 2.52, 2.53, 2.54, 2.55, 2.56, 2.57, 2.58, 2.59,
        2.60, 2.61, 2.62, 2.63, 2.64, 2.65, 2.66, 2.67, 2.68, 2.69,
        2.70, 2.71, 2.72, 2.73, 2.74, 2.75, 2.76, 2.77, 2.78, 2.79,
        2.80, 2.81, 2.82, 2.83, 2.84, 2.85, 2.86, 2.87, 2.88, 2.89,
        2.90, 2.91, 2.92, 2.93, 2.94, 2.95, 2.96, 2.97, 2.98, 2.99,
        3.00, 3.01, 3.02, 3.03, 3.04, 3.05, 3.06, 3.07, 3.08, 3.09,
        3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19,
        3.20, 3.21, 3.22, 3.23, 3.24, 3.25, 3.26, 3.27, 3.28, 3.29,
        3.30, 3.31, 3.32, 3.33, 3.34, 3.35, 3.36, 3.37, 3.38, 3.39,
        3.40, 3.41, 3.42, 3.43, 3.44, 3.45, 3.46, 3.47, 3.48, 3.49,
        3.50, 3.51, 3.52, 3.53, 3.54, 3.55, 3.56, 3.57, 3.58, 3.59,
        3.60, 3.61, 3.62, 3.63, 3.64, 3.65, 3.66, 3.67, 3.68, 3.69,
        3.70, 3.71, 3.72, 3.73, 3.74, 3.75, 3.76, 3.77, 3.78, 3.79,
        3.80, 3.81, 3.82, 3.83, 3.84, 3.85, 3.86, 3.87, 3.88, 3.89,
        3.90, 3.91, 3.92, 3.93, 3.94, 3.95, 3.96, 3.97, 3.98, 3.99,
        4.00, 4.01, 4.02, 4.03, 4.04, 4.05, 4.06, 4.07, 4.08, 4.09,
        4.10, 4.11, 4.12, 4.13, 4.14, 4.15, 4.16, 4.17, 4.18, 4.19,
        4.20, 4.21, 4.22, 4.23, 4.24, 4.25, 4.26, 4.27, 4.28, 4.29,
        4.30, 4.31, 4.32, 4.33, 4.34, 4.35, 4.36, 4.37, 4.38, 4.39,
        4.40, 4.41, 4.42, 4.43, 4.44, 4.45, 4.46, 4.47, 4.48, 4.49,
        4.50, 4.51, 4.52, 4.53, 4.54, 4.55, 4.56, 4.57, 4.58, 4.59,
        4.60, 4.61, 4.62, 4.63, 4.64, 4.65, 4.66, 4.67, 4.68, 4.69,
        4.70, 4.71, 4.72, 4.73, 4.74, 4.75, 4.76, 4.77, 4.78, 4.79,
        4.80, 4.81, 4.82, 4.83, 4.84, 4.85, 4.86, 4.87, 4.88, 4.89,
        4.90, 4.91, 4.92, 4.93, 4.94, 4.95, 4.96, 4.97, 4.98, 4.99,
        5.00, 5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07, 5.08, 5.09,
        5.10, 5.11, 5.12, 5.13, 5.14, 5.15, 5.16, 5.17, 5.18, 5.19,
        5.20, 5.21, 5.22
    ])

    # Vapor Pressure P from Table I (in 1.e-3 mmHg)
    p_data = np.array([
        0.016342, 0.022745, 0.031287, 0.042561, 0.057292, 0.076356, 0.10081, 0.13190, 0.17112, 0.22021,
        0.28121, 0.35649, 0.44877, 0.56118, 0.69729, 0.86116, 1.0574, 1.2911, 1.5682, 1.8949,
        2.2787, 2.7272, 3.2494, 3.8549, 4.5543, 5.3591, 6.2820, 7.3365, 8.5376, 9.9013,
        11.445, 13.187, 15.147, 17.348, 19.811, 22.561, 25.624, 29.027, 32.800, 36.974,
        41.581, 46.656, 52.234, 58.355, 65.059, 72.386, 80.382, 89.093, 98.567, 108.853,
        120.000, 132.070, 145.116, 159.198, 174.375, 190.711, 208.274, 227.132, 247.350, 269.006,
        292.169, 316.923, 343.341, 371.512, 401.514, 433.437, 467.365, 503.396, 541.617, 582.129,
        625.025, 670.411, 718.386, 769.057, 822.527, 878.916, 938.330, 1000.87, 1066.67, 1135.85,
        1208.51, 1284.81, 1364.83, 1448.73, 1536.61, 1628.62, 1724.91, 1825.58, 1930.79, 2040.67,
        2155.35, 2274.99, 2399.73, 2529.72, 2665.09, 2805.99, 2952.60, 3105.04, 3263.48, 3428.07,
        3598.97, 3776.32, 3960.32, 4151.07, 4348.79, 4553.58, 4765.68, 4985.18, 5212.26, 5447.11,
        5689.88, 5940.76, 6199.90, 6467.42, 6743.57, 7028.47, 7322.31, 7625.21, 7937.40, 8259.02,
        8590.22, 8931.18, 9282.06, 9643.02, 10014.3, 10395.9, 10788.2, 11191.2, 11605.1, 12030.1,
        12466.1, 12913.7, 13372.8, 13843.6, 14326.1, 14820.7, 15327.3, 15846.3, 16377.7, 16921.7,
        17478.2, 18047.7, 18630.1, 19225.5, 19834.1, 20455.9, 21091.1, 21739.7, 22402.0, 23077.9,
        23767.4, 24470.9, 25188.1, 25919.2, 26664.2, 27423.3, 28196.3, 28983.2, 29784.2, 30599.1,
        31428.1, 32271.1, 33128.0, 33998.6, 34882.8, 35780.3, 36690.9, 37614.3, 38550.2, 39500.3,
        40465.6, 41446.6, 42443.5, 43456.5, 44485.7, 45531.3, 46593.5, 47672.5, 48768.6, 49881.8,
        51012.3, 52160.2, 53325.8, 54509.2, 55710.5, 56930.0, 58167.8, 59423.8, 60698.8, 61992.0,
        63304.3, 64635.2, 65985.4, 67354.8, 68743.5, 70152.0, 71580.2, 73028.1, 74496.0, 75984.2,
        77493.1, 79022.2, 80572.2, 82142.9, 83734.6, 85347.2, 86981.2, 88636.7, 90313.8, 92012.6,
        93733.4, 95476.0, 97240.8, 99028.2, 100838, 102669, 104525, 106403, 108304, 110228,
        112175, 114145, 116139, 118156, 120198, 122263, 124353, 126465, 128603, 130765,
        132952, 135164, 137401, 139663, 141949, 144260, 146597, 148961, 151349, 153763,
        156204, 158671, 161164, 163684, 166230, 168802, 171402, 174028, 176682, 179364,
        182073, 184810, 187574, 190366, 193187, 196037, 198914, 201820, 204755, 207719,
        210711, 213732, 216783, 219864, 222975, 226115, 229285, 232484, 235714, 238974,
        242266, 245587, 248939, 252322, 255736, 259182, 262658, 266166, 269706, 273278,
        276880, 280516, 284183, 287883, 291615, 295380, 299178, 303008, 306871, 310768,
        314697, 318659, 322654, 326684, 330747, 334845, 338976, 343141, 347341, 351575,
        355844, 360147, 364485, 368860, 373269, 377714, 382194, 386710, 391262, 395849,
        400471, 405130, 409825, 414556, 419324, 424128, 428968, 433846, 438760, 443713,
        448702, 453729, 458794, 463897, 469038, 474218, 479435, 484691, 489985, 495317,
        500688, 506098, 511547, 517036, 522564, 528132, 533739, 539387, 545075, 550805,
        556574, 562383, 568234, 574126, 580059, 586034, 592051, 598110, 604210, 610352,
        616537, 622764, 629033, 635345, 641700, 648099, 654541, 661026, 667554, 674125,
        680740, 687399, 694103, 700851, 707643, 714479, 721360, 728285, 735255, 742269,
        749328, 756431, 763579, 770772, 778010, 785294, 792623, 799999, 807422, 814893,
        822411, 829978, 837592, 845255, 852966, 860725, 868533, 876390, 884296, 892252,
        900258, 908313, 916418, 924573, 932778, 941033, 949338, 957693, 966099, 974556,
        983066, 991628, 1000239, 1008905, 1017621, 1026390, 1035213, 1044087, 1053014, 1061995,
        1071029, 1080114, 1089254, 1098449, 1107699, 1117002, 1126359, 1135772, 1145239, 1154761,
        1164339, 1173972, 1183662, 1193407, 1203209, 1213066, 1222981, 1232955, 1242983, 1253069,
        1263212, 1273414, 1283673, 1293991, 1304367, 1314802, 1325297, 1335850, 1346462, 1357136,
        1367870, 1378662, 1389516, 1400429, 1411404, 1422438, 1433533, 1444690, 1455911, 1467191,
        1478535, 1489940, 1501409, 1512940, 1524535, 1536192, 1547912, 1559698, 1571546, 1583458,
        1595437, 1607481, 1619589, 1631761, 1644000, 1656305, 1668673, 1681108, 1693612, 1706180,
        1718817, 1731521, 1744290
    ])
    return CubicSpline(t_data, p_data)

def p_He58(T58):
    """
    Calculates the vapor pressure of 4He using cubic spline interpolation 
    based on the 1958 He Scale of Temperatures. Taken from Table I of H. Van Dijk et al.,
    "Vapour Pressure Temperature Scale for the Liquid 4He Region," Physica 24, S129-S131 (1958)
    
    Units: 
    - T58: K
    - Output: Pa
    """
    
    # Boundary check per the 1958 Scale limits (0.5K to 5.22K)
    if T58 < 0.5 or T58 > 5.22:
        raise ValueError("T58 must be between 0.5 and 5.22 K.")

    # Instantiate the cubic spline interpolation
    cs = get_He58_spline()
    
    p_mmHg = float(cs(T58)) * 1.e-3
# Convert mmHg to Pascals
    return p_mmHg /760.0 * 101325.0

@lru_cache(maxsize=1)
def get_He58_tspline():
    """
    Defines the data and fits the spline only once for 1958 He vapor pressure
    Taken from Table I of H. Van Dijk et al.,
    "Vapour Pressure Temperature Scale for the Liquid 4He Region," Physica 24, S129-S131 (1958)
    This time for p(T) instead of T(p)
    """
    # Temperature T in Kelvin (0.50 to 5.22)
    t_data = np.array([
        0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
        0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
        0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
        0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
        0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09,
        1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19,
        1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29,
        1.30, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39,
        1.40, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49,
        1.50, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59,
        1.60, 1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69,
        1.70, 1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79,
        1.80, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89,
        1.90, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99,
        2.00, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08, 2.09,
        2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19,
        2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28, 2.29,
        2.30, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39,
        2.40, 2.41, 2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48, 2.49,
        2.50, 2.51, 2.52, 2.53, 2.54, 2.55, 2.56, 2.57, 2.58, 2.59,
        2.60, 2.61, 2.62, 2.63, 2.64, 2.65, 2.66, 2.67, 2.68, 2.69,
        2.70, 2.71, 2.72, 2.73, 2.74, 2.75, 2.76, 2.77, 2.78, 2.79,
        2.80, 2.81, 2.82, 2.83, 2.84, 2.85, 2.86, 2.87, 2.88, 2.89,
        2.90, 2.91, 2.92, 2.93, 2.94, 2.95, 2.96, 2.97, 2.98, 2.99,
        3.00, 3.01, 3.02, 3.03, 3.04, 3.05, 3.06, 3.07, 3.08, 3.09,
        3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19,
        3.20, 3.21, 3.22, 3.23, 3.24, 3.25, 3.26, 3.27, 3.28, 3.29,
        3.30, 3.31, 3.32, 3.33, 3.34, 3.35, 3.36, 3.37, 3.38, 3.39,
        3.40, 3.41, 3.42, 3.43, 3.44, 3.45, 3.46, 3.47, 3.48, 3.49,
        3.50, 3.51, 3.52, 3.53, 3.54, 3.55, 3.56, 3.57, 3.58, 3.59,
        3.60, 3.61, 3.62, 3.63, 3.64, 3.65, 3.66, 3.67, 3.68, 3.69,
        3.70, 3.71, 3.72, 3.73, 3.74, 3.75, 3.76, 3.77, 3.78, 3.79,
        3.80, 3.81, 3.82, 3.83, 3.84, 3.85, 3.86, 3.87, 3.88, 3.89,
        3.90, 3.91, 3.92, 3.93, 3.94, 3.95, 3.96, 3.97, 3.98, 3.99,
        4.00, 4.01, 4.02, 4.03, 4.04, 4.05, 4.06, 4.07, 4.08, 4.09,
        4.10, 4.11, 4.12, 4.13, 4.14, 4.15, 4.16, 4.17, 4.18, 4.19,
        4.20, 4.21, 4.22, 4.23, 4.24, 4.25, 4.26, 4.27, 4.28, 4.29,
        4.30, 4.31, 4.32, 4.33, 4.34, 4.35, 4.36, 4.37, 4.38, 4.39,
        4.40, 4.41, 4.42, 4.43, 4.44, 4.45, 4.46, 4.47, 4.48, 4.49,
        4.50, 4.51, 4.52, 4.53, 4.54, 4.55, 4.56, 4.57, 4.58, 4.59,
        4.60, 4.61, 4.62, 4.63, 4.64, 4.65, 4.66, 4.67, 4.68, 4.69,
        4.70, 4.71, 4.72, 4.73, 4.74, 4.75, 4.76, 4.77, 4.78, 4.79,
        4.80, 4.81, 4.82, 4.83, 4.84, 4.85, 4.86, 4.87, 4.88, 4.89,
        4.90, 4.91, 4.92, 4.93, 4.94, 4.95, 4.96, 4.97, 4.98, 4.99,
        5.00, 5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07, 5.08, 5.09,
        5.10, 5.11, 5.12, 5.13, 5.14, 5.15, 5.16, 5.17, 5.18, 5.19,
        5.20, 5.21, 5.22
    ])

    # Vapor Pressure P from Table I (in 1.e-3 mmHg)
    p_data = np.array([
        0.016342, 0.022745, 0.031287, 0.042561, 0.057292, 0.076356, 0.10081, 0.13190, 0.17112, 0.22021,
        0.28121, 0.35649, 0.44877, 0.56118, 0.69729, 0.86116, 1.0574, 1.2911, 1.5682, 1.8949,
        2.2787, 2.7272, 3.2494, 3.8549, 4.5543, 5.3591, 6.2820, 7.3365, 8.5376, 9.9013,
        11.445, 13.187, 15.147, 17.348, 19.811, 22.561, 25.624, 29.027, 32.800, 36.974,
        41.581, 46.656, 52.234, 58.355, 65.059, 72.386, 80.382, 89.093, 98.567, 108.853,
        120.000, 132.070, 145.116, 159.198, 174.375, 190.711, 208.274, 227.132, 247.350, 269.006,
        292.169, 316.923, 343.341, 371.512, 401.514, 433.437, 467.365, 503.396, 541.617, 582.129,
        625.025, 670.411, 718.386, 769.057, 822.527, 878.916, 938.330, 1000.87, 1066.67, 1135.85,
        1208.51, 1284.81, 1364.83, 1448.73, 1536.61, 1628.62, 1724.91, 1825.58, 1930.79, 2040.67,
        2155.35, 2274.99, 2399.73, 2529.72, 2665.09, 2805.99, 2952.60, 3105.04, 3263.48, 3428.07,
        3598.97, 3776.32, 3960.32, 4151.07, 4348.79, 4553.58, 4765.68, 4985.18, 5212.26, 5447.11,
        5689.88, 5940.76, 6199.90, 6467.42, 6743.57, 7028.47, 7322.31, 7625.21, 7937.40, 8259.02,
        8590.22, 8931.18, 9282.06, 9643.02, 10014.3, 10395.9, 10788.2, 11191.2, 11605.1, 12030.1,
        12466.1, 12913.7, 13372.8, 13843.6, 14326.1, 14820.7, 15327.3, 15846.3, 16377.7, 16921.7,
        17478.2, 18047.7, 18630.1, 19225.5, 19834.1, 20455.9, 21091.1, 21739.7, 22402.0, 23077.9,
        23767.4, 24470.9, 25188.1, 25919.2, 26664.2, 27423.3, 28196.3, 28983.2, 29784.2, 30599.1,
        31428.1, 32271.1, 33128.0, 33998.6, 34882.8, 35780.3, 36690.9, 37614.3, 38550.2, 39500.3,
        40465.6, 41446.6, 42443.5, 43456.5, 44485.7, 45531.3, 46593.5, 47672.5, 48768.6, 49881.8,
        51012.3, 52160.2, 53325.8, 54509.2, 55710.5, 56930.0, 58167.8, 59423.8, 60698.8, 61992.0,
        63304.3, 64635.2, 65985.4, 67354.8, 68743.5, 70152.0, 71580.2, 73028.1, 74496.0, 75984.2,
        77493.1, 79022.2, 80572.2, 82142.9, 83734.6, 85347.2, 86981.2, 88636.7, 90313.8, 92012.6,
        93733.4, 95476.0, 97240.8, 99028.2, 100838, 102669, 104525, 106403, 108304, 110228,
        112175, 114145, 116139, 118156, 120198, 122263, 124353, 126465, 128603, 130765,
        132952, 135164, 137401, 139663, 141949, 144260, 146597, 148961, 151349, 153763,
        156204, 158671, 161164, 163684, 166230, 168802, 171402, 174028, 176682, 179364,
        182073, 184810, 187574, 190366, 193187, 196037, 198914, 201820, 204755, 207719,
        210711, 213732, 216783, 219864, 222975, 226115, 229285, 232484, 235714, 238974,
        242266, 245587, 248939, 252322, 255736, 259182, 262658, 266166, 269706, 273278,
        276880, 280516, 284183, 287883, 291615, 295380, 299178, 303008, 306871, 310768,
        314697, 318659, 322654, 326684, 330747, 334845, 338976, 343141, 347341, 351575,
        355844, 360147, 364485, 368860, 373269, 377714, 382194, 386710, 391262, 395849,
        400471, 405130, 409825, 414556, 419324, 424128, 428968, 433846, 438760, 443713,
        448702, 453729, 458794, 463897, 469038, 474218, 479435, 484691, 489985, 495317,
        500688, 506098, 511547, 517036, 522564, 528132, 533739, 539387, 545075, 550805,
        556574, 562383, 568234, 574126, 580059, 586034, 592051, 598110, 604210, 610352,
        616537, 622764, 629033, 635345, 641700, 648099, 654541, 661026, 667554, 674125,
        680740, 687399, 694103, 700851, 707643, 714479, 721360, 728285, 735255, 742269,
        749328, 756431, 763579, 770772, 778010, 785294, 792623, 799999, 807422, 814893,
        822411, 829978, 837592, 845255, 852966, 860725, 868533, 876390, 884296, 892252,
        900258, 908313, 916418, 924573, 932778, 941033, 949338, 957693, 966099, 974556,
        983066, 991628, 1000239, 1008905, 1017621, 1026390, 1035213, 1044087, 1053014, 1061995,
        1071029, 1080114, 1089254, 1098449, 1107699, 1117002, 1126359, 1135772, 1145239, 1154761,
        1164339, 1173972, 1183662, 1193407, 1203209, 1213066, 1222981, 1232955, 1242983, 1253069,
        1263212, 1273414, 1283673, 1293991, 1304367, 1314802, 1325297, 1335850, 1346462, 1357136,
        1367870, 1378662, 1389516, 1400429, 1411404, 1422438, 1433533, 1444690, 1455911, 1467191,
        1478535, 1489940, 1501409, 1512940, 1524535, 1536192, 1547912, 1559698, 1571546, 1583458,
        1595437, 1607481, 1619589, 1631761, 1644000, 1656305, 1668673, 1681108, 1693612, 1706180,
        1718817, 1731521, 1744290
    ])
    return CubicSpline(p_data, t_data)

def t_He58(p58):
    """
    Calculates the temperature from the vapor pressure of 4He using cubic spline interpolation 
    based on the 1958 He Temperature Scale. Taken from Table I of H. Van Dijk et al.,
    "Vapour Pressure Temperature Scale for the Liquid 4He Region," Physica 24, S129-S131 (1958)
    
    Units: 
    - p58: Pa
    - Output: K
    """
    plow = 0.016342 / 1000 / 760.0 * 101325.0  # Convert lower bound from micro-mHg to Pascals
    phigh = 1744290 / 1000 / 760.0 * 101325.0  # Convert upper bound from micro-mHg to Pascals
    # Boundary check per the 1958 Scale limits (0.5 K to 5.22 K)
    if p58 < plow or p58 > phigh:
        raise ValueError(f"In t_He58(p58), p58 must be between {plow:.8f} and {phigh:.8f} Pa.")

    # cubic spline interpolation
    cs = get_He58_tspline()
    
    p_microns = p58 * 760.0 / 101325.0 * 1e3  # Convert Pascals to micro-mHg
    T58 = float(cs(p_microns))
# Convert mmHg to Pascals
    return T58

def p_3He62(T62):
    """
    Calculates the vapor pressure of 3He 
    based on the 1962 3He Scale of Temperatures. Taken from S.G. Sydoriak et al.,
    "The 1962 He^3 Scale of Temperatures. II. Derivation," J. Res. NBS 68A, 559-565 (1964)
    
    Units: 
    - T62: K
    - Output: Pa
    """
    
    # Boundary check per the 1962 Scale limits (0.2 K to 2 K)
    if T62 < 0.2 or T62 > 3.324:
        raise ValueError("T62 must be between 0.2 and 3.324 K.")

    # Evaluate equation
    Ai = np.array([4.80386, -0.286001, 0.198608, -0.0502237, 0.00505486])
    poly = Polynomial(Ai)
    lnp = poly(T62) - 2.49174/T62 + 2.24846*np.log(T62)
    p_mmHg = np.exp(lnp)
    
# Convert mmHg to Pascals
    return p_mmHg /760.0 * 101325.0

def p_He90(T90):
    """
    Calculates the vapor pressure of 4He using the defining equation for the
    ITS-90 temperature scale from H. Preston-Thomas, "The International Temperature
    Scale of 1990 (ITS-90)," Metrologia 27, 3-10 (1990)
    Solves iteratuvely using ITS-90 defining equations from 1.25 to 5 K
    
    Units: 
    - T90: K
    - Output: Pa
    """
    T_lambda = 2.1768
    if T90 < 1.25 or T90 > 5.0:
        raise ValueError("T90 must be between 1.25 and 5.0 K.")
    elif T90 < T_lambda:
# Below lambda point
        plow = 114.734339
        lnplow = np.log(plow)
        phigh = 5041.815  
        lnphigh = np.log(phigh) 
        Tlow = t_He90(plow)
        Thigh = t_He90(phigh)
        p_old = plow
        while True:
            frac = (T90 - Tlow) / (Thigh - Tlow)
            lnp_new = lnplow + frac * (lnphigh - lnplow)
            p_new = np.exp(lnp_new)
            if abs((p_new - p_old)/p_new) < 1e-9:
                return p_new
            else:
                T_new = t_He90(p_new)
                if T_new > T90:
                    Thigh = T_new
                    phigh = p_new
                    p_old = p_new
                    lnphigh = np.log(phigh)
                else:
                    Tlow = T_new
                    plow = p_new
                    p_old = p_new
                    lnplow = np.log(plow)
        
    else:
# Above lambda point
        plow = 5041.815
        Tlow = t_He90(plow)
        lnplow = np.log(plow)
        phigh = 196016.5329  
        Thigh = t_He90(phigh)
        lnphigh = np.log(phigh) 
        p_old = plow
        while True:
            frac = (T90 - Tlow) / (Thigh - Tlow)
            lnp_new = lnplow + frac * (lnphigh - lnplow)
            p_new = np.exp(lnp_new)
            if abs((p_new - p_old)/p_new) < 1e-11:
                return p_new
            else:
                T_new = t_He90(p_new)
                if T_new > T90:
                    Thigh = T_new
                    phigh = p_new
                    p_old = p_new
                    lnphigh = np.log(phigh)
                else:
                    Tlow = T_new
                    plow = p_new
                    p_old = p_new
                    lnplow = np.log(plow)

def t_He90(p):
    """
    Calculates the ITS-90 temperature based on the vapor pressure of He-4 using Eq. (3)
    from H. Preston-Thomas, "The International Temperature
    Scale of 1990 (ITS-90)," Metrologia 27, 3-10 (1990)
    Note different equations above and below lambda-point temperature of 2.1768 K.

    Units: 
    - p: Pa
    - Output: K
    """

    plow = 114.734339
    phigh = 196016.5329
    plambda = 5041.815
    if not plow <= p <= phigh:
        raise ValueError(f"Pressure must be between {plow} and {phigh} Pascals.")  
    elif  p <= plambda:
        # Below lambda point
        Ai = np.array([1.392408, 0.527153, 0.166756, 0.050988, 0.026514,
                      0.001975, -0.017976, 0.005409, 0.013259])
        B = 5.6
        C = 2.9
        frac = (np.log(p)-B) / C
        poly = Polynomial(Ai)
        return poly(frac)
    else:
        Ai = np.array([3.146631, 1.357655, 0.413923, 0.091159, 0.016349,
                       0.001826, -0.004325, -0.004973])
        B = 10.3
        C = 1.9
        frac = (np.log(p)-B) / C
        poly = Polynomial(Ai)
        return poly(frac)

def p_3He90(T90):
    """
    Calculates the vapor pressure of 3He using the defining equation for the
    ITS-90 temperature scale from H. Preston-Thomas, "The International Temperature
    Scale of 1990 (ITS-90)," Metrologia 27, 3-10 (1990)
    Solves iteratuvely using ITS-90 defining equation from 0.65 to 3.2 K
    
    Units: 
    - T90: K
    - Output: Pa
    """
    if T90 < 0.65 or T90 > 3.2:
        raise ValueError("T90 must be between 0.65 and 3.2 K.")
    else:
# Below lambda point
        plow = 115.905619
        lnplow = np.log(plow)
        phigh = 101662.101  
        lnphigh = np.log(phigh) 
        Tlow = t_3He90(plow)
        Thigh = t_3He90(phigh)
        p_old = plow
        while True:
            frac = (T90 - Tlow) / (Thigh - Tlow)
            lnp_new = lnplow + frac * (lnphigh - lnplow)
            p_new = np.exp(lnp_new)
            if abs((p_new - p_old)/p_new) < 1e-9:
                return p_new
            else:
                T_new = t_3He90(p_new)
                if T_new > T90:
                    Thigh = T_new
                    phigh = p_new
                    p_old = p_new
                    lnphigh = np.log(phigh)
                else:
                    Tlow = T_new
                    plow = p_new
                    p_old = p_new
                    lnplow = np.log(plow)

def t_3He90(p):
    """
    Calculates the ITS-90 temperature based on the vapor pressure of He-3 using Eq. (3)
    from H. Preston-Thomas, "The International Temperature
    Scale of 1990 (ITS-90)," Metrologia 27, 3-10 (1990)
    
    Units: 
    - p: Pa
    - Output: K
    """

    plow = 115.905619 # pressure at 0.65 K
    phigh = 101662.101 # pressure at 3.2 K
    if not plow <= p <= phigh:
        raise ValueError(f"Pressure must be between {plow} and {phigh} Pascals.")  
    else:
        Ai = np.array([1.053447, 0.980106, 0.676380, 0.372692, 0.151656,
                        -0.002263, 0.006596, 0.088966, -0.004770, -0.054943])
        B = 7.3
        C = 4.3
        frac = (np.log(p)-B) / C
        poly = Polynomial(Ai)
        return poly(frac)

def p_PTB2006(T2006):    
    """
    Calculates the vapor pressure of He-3 on the PTB-2006 scale
    using Eq. (8) from J. Engert et al., Metrologia 44, 40-52 (2007)
     
    Units: 
    - T2006: K
    - Output: Pa
    """
# Boundary checking, although this should already be done in calling routine t90_fromPTB2006
    if T2006 < 0.63 or T2006 > 3.25:
        raise ValueError("T2006 must be between 0.63 and 3.25 K.")

# Equation (8) coefficients (a_i)
# Note: a_{-1} is index 0, a_0 is index 1, ..., a_5 is index 6
    A_COEFFS = {
    -1: -2.49255841,
     0: 9.55598062,
     1: -0.10775789,
     2: 0.13899533,
     3: -0.03962933,
     4: 0.00432461,
     5: 2.16565173
    }
    poly_sum = 0
    for i in range(-1, 5):
        poly_sum += A_COEFFS[i] * (T2006 ** i)
    
    # ln(p/Pa) = sum + a_5 * ln(T/K) [cite: 633]
    ln_p = poly_sum + A_COEFFS[5] * np.log(T2006)
    return np.exp(ln_p)

    
def t76_from_NBS220(T220):
    """
    Calculates the EPT-76 temperature from the NBS 2-20 scale, using conversion table from
    the EPT-76 paper "The 1976 Provisional 0.5 K to 30 K Temperature Scale," 
    Metrologia, 15, 65-68 (1979). Differences look too irregular for cubic spline,
    so interpolate linearly instead.
    NBS 2-20 is defined from 2.3 K to 20 K in H. Plumb and G. Cataland,
    Metrologia 2, 127-139 (1966)
    
    Units: 
    - T220: K
    - Output: K
    """
    if T220 < 2.3 or T220 > 20.0:
        raise ValueError("NBS 2-20 temperature must be between 2.3 and 20 K.")
    else:
    # linear interpolation of differences in mK between T220 and T76 from the table in the EPT-76 paper
        T_data = np.array([
            2.3, 2.8, 3.2, 4.2, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
            ])
        diff_data = np.array([
            2.1, -1.1, 1.0, 2.5, 3.0, 1.7, 4.8, 2.2, -1.5, -2.1, -1.0,
            0.2, -1.8, -2.2, -0.6, 0.9, 1.7, -0.9, -0.2, -0.8
            ])
        diff = np.interp(T220, T_data, diff_data)
        return T220 - diff * 1e-3

@lru_cache(maxsize=1)
def get_NBS55_spline():
    """
    Spline for converting NBS-55 to IPTS-68, based on data from 
    R.E. Bedford et al., Metrologia 5, 47-49 (1969).
    """
# Data from Table 2: Relationships Between IPTS-68 and NBS-55 
# x = T68 (Kelvins)
# y = IPTS-68 minus NBS-55 (mK)
# Add point at the equilibrium H2 triple point
    x_points = np.array([13.81,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 83, 84, 85, 86, 87, 88, 89, 90, 91
    ])

    y_points = np.array([-3.5,
    -2.3, # 14 K
    2.0, 5.1, 7.1, 8.4, 8.8, 9.0, 8.9, 8.8, 8.6, 8.3, 7.8, 7.3, 7.1, 7.1, 7.3, 7.6, # 15-30 K
    8.0, 8.4, 9.0, 9.9, 11.0, 12.2, 13.3, 14.2, 14.9, 15.4, 15.7, 15.8, 15.7, 15.5, 15.1, # 31-45 K
    14.8, 14.5, 14.2, 13.7, 13.1, 12.4, 11.6, 10.4, 8.9, 7.1, 5.2, 3.4, 1.7, 0.3, # 46-59 K
    -0.8, -1.4, -1.5, -1.2, -0.7, -0.1, 0.5, 0.9, 1.1, 0.9, 0.3, -0.6, -1.7, # 60-72 K
    -3.0, -4.3, -5.6, -6.8, -7.8, -8.6, -9.0, -9.0, -8.6, -7.7, -6.4, -4.9, -2.9, -0.5, # 73-86 K
    2.2, 4.9, 7.4, 9.6, 11.1 # 87-91 K
    ])
    return CubicSpline(x_points, y_points)

def t68_from_NBS55(T55):
    """
    Calculates the IPTS-68 temperature from the NBS-55 scale.
    Mostly uses conversion from R.E. Bedford et al., Metrologia 5, 47-49 (1969).
    Can use this source to do cubic spline interpolation from H2 TP to 91 K.
    
    Units: 
    - T55: K
    - Output: K
    """
    if T55 < 13.8135 or T55 > 91:
        raise ValueError("NBS55 temperature must be between 13.8165 and 91 K.")
    else:
# Since table is f(T68) instead of f(T55), need to iterate to converge on T68
        T_old = T55

# Initialize Cubic Spline interpolation
        cs = get_NBS55_spline()

        while True:
            diff = cs(T_old)
            T_new = T55 + diff * 1e-3   
            if abs(T_new - T_old) < 1e-12:
                return T_new
            else:
                T_old = T_new

@lru_cache(maxsize=1)
def get_NPL61_spline():
    """
    Spline for converting NPL-61 to IPTS-68, based on data from 
    R.E. Bedford et al., Metrologia 5, 47-49 (1969).
    """
    # Data from Table 1 and Table 2: Relationships Between IPTS-68 and NPL-61 
    # x = T68 (Kelvins)
    # y = IPTS-68 minus NPL-61 (mK)
    # x points include the equilibrium H2 triple point (13.81 K) and 14-91 K
    x_points = np.array([
        13.81, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91
    ])

    # y points (diff in mK) from Table 2
    # Note: 13.81 K value is derived from Table 1 (13.81 - 13.8129 = -2.9 mK) 
    y_points = np.array([
        -2.9,  # 13.81 K (Derived from Table 1: 13.81 - 13.8129)
        1.4, 2.8, 1.3, 0.1, 4.4, 9.9, 9.7, 7.3, 5.3, 4.3, 3.7, 4.1, 4.5, 5.0, 5.4, 5.3, 4.9, # 14-30 K 
        4.0, 2.9, 1.6, 0.5, # 31-34 K 
        -0.5, -1.1, -1.5, -1.5, -1.3, -0.9, -0.5, -0.3, -0.3, -0.6, -1.1, # 35-45 K (Negative Range) 
        -1.7, -2.4, -3.2, -4.0, -4.6, -4.9, -4.8, -4.7, -4.1, -3.2, -2.0, -0.5, # 46-57 K (Negative Range) 
        1.1, 2.8, 4.3, 5.9, 7.2, 8.1, 8.8, 9.2, 9.1, 8.8, 8.2, 7.3, 6.3, 5.2, 4.0, # 58-72 K 
        3.0, 2.2, 1.6, 1.3, 1.3, 1.7, 2.4, 3.4, 4.6, 5.9, 7.3, 8.5, 9.5, 10.2, 10.5, 10.5, 10.1, 9.3, 7.9 # 73-91 K
    ])
    return CubicSpline(x_points, y_points)

def t68_from_NPL61(T61):
    """
    Calculates the IPTS-68 temperature from the NPL-61 scale.
    Data sourced from R.E. Bedford et al., Metrologia 5, 47-49 (1969).
    Uses iterative cubic spline interpolation to find T68 from T61.
    
    Units: 
    - T61: K
    - Output: K (IPTS-68)
    """
    # Range check based on NPL-61 values at the H2 triple point (13.8129 K) 
    # and the upper limit of Table 2 (91 K).
    if T61 < 13.8129 or T61 > 91:
        raise ValueError("NPL-61 temperature must be between 13.8129 and 91 K.")

    # Initialize Cubic Spline interpolation
    cs = get_NPL61_spline()

    # Iterative solver: T68 = T61 + Delta_T(T68)
    T_old = T61
    while True:
        diff = cs(T_old)
        T_new = T61 + diff * 1e-3   
        if abs(T_new - T_old) < 1e-12:
            return T_new
        else:
            T_old = T_new

@lru_cache(maxsize=1)
def get_PRMI54_spline():
    """
    Spline for converting PRMI-54 to IPTS-68, based on data from 
    R.E. Bedford et al., Metrologia 5, 47-49 (1969).
    """
   # x points: T68 (Kelvins) from Table 1 and Table 2 
    x_points = np.array([
        13.81, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 83, 84, 85, 86, 87, 88, 89, 90, 91
    ])

    # y points: IPTS-68 minus PRMI-54 (mK)
    y_points = np.array([
        -49.0, # 13.81 K (From Table 1: 13.81 - 13.8590) 
        -47.4, -36.9, -28.0, -21.0, -16.5, -14.2, -12.9, -11.9, -10.8, 
        -9.7, -9.1, -8.6, -8.2, -7.8, -7.4, -7.2, -7.3, # 23-30 K 
        -7.5, -7.6, -7.6, -7.6, -7.5, -7.3, -7.3, -7.3, -7.0, # 31-39 K 
        -7.0, -7.0, -6.9, -6.9, -6.9, -6.7, -6.6, -6.1, -5.8, -5.3, # 40-49 K 
        -4.8, -4.2, -3.8, -3.5, -3.4, -3.4, -3.6, -4.3, -5.3, -6.5, # 50-59 K
        -7.9, -9.3, -10.6, -12.0, -13.1, -14.0, -14.5, -15.0, -15.2, -15.2, # 60-69 K
        -15.0, -15.0, -14.8, -14.7, -14.7, -14.6, -14.5, -14.4, -14.3, -14.4, # 70-79 K
        -14.5, -14.7, -14.9, -15.0, -14.9, -14.8, -14.5, -14.0, -13.4, -13.0, -12.7, -12.7 # 80-91 K
    ])
    return CubicSpline(x_points, y_points)

def t68_from_PRMI54(TPRM):
    """
    Calculates the IPTS-68 temperature from the PRMI-54 scale.
    Data sourced from R.E. Bedford et al., Metrologia 5, 47-49 (1969).
    Uses iterative cubic spline interpolation to find T68 from TPRM.
    
    Units: 
    - TPRM: K (PRMI-54)
    - Output: K (IPTS-68)
    """
    # Range check based on PRMI-54 values at the H2 triple point (13.8590 K) 
    # and the upper limit of Table 2 (approx 91 K).
    if TPRM < 13.8590 or TPRM > 91:
        raise ValueError("PRMI-54 temperature must be between 13.8590 and 91 K.")
    
    # Initialize Cubic Spline interpolation
    cs = get_PRMI54_spline()

    # Iterative solver: T68 = TPRM + Delta_T(T68)
    T_old = TPRM
    while True:
        diff = cs(T_old)
        T_new = TPRM + diff * 1e-3   
        if abs(T_new - T_old) < 1e-12:
            return T_new
        else:
            T_old = T_new

@lru_cache(maxsize=1)
def get_PSU54_spline():
    """
    Spline for converting PSU-54 to IPTS-68, based on data from 
    R.E. Bedford et al., Metrologia 5, 47-49 (1969).
    """
    # x points: T68 (Kelvins) from Table 1  and Table 2 
    x_points = np.array([
        13.81, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 83, 84, 85, 86, 87, 88, 89, 90, 91
    ])

    # y points: IPTS-68 minus PSU-54 (mK)
    y_points = np.array([
        9.0,   # 13.81 K (From Table 1: 13.81 - 13.8010 )
        9.9, 16.4, 19.3, 21.4, 22.4, 22.9, 23.3, 23.5, 23.7, 24.0, 24.6, 24.6, 24.1, 23.5, 23.4, 23.6, 24.0, # 14-30 K
        24.5, 25.3, 27.0, 28.5, 28.8, 28.8, 28.8, 28.8, 28.7, 28.6, 28.1, 27.5, 27.0, 26.7, 26.7,            # 31-45 K
        26.4, 25.7, 24.7, 23.2, 21.4, 20.7, 20.6, 20.7, 20.6, 20.3, 19.9, 19.0, 18.3, 18.1,                  # 46-59 K
        18.0, 18.0, 18.0, 18.0, 18.1, 18.8, 18.9, 19.6, 20.4, 21.7, 22.6, 22.8, 22.5, 22.0, 21.2, 20.9,      # 60-75 K
        21.7, 22.6, 23.3, 23.9, 24.4, 25.2, 26.2, 27.2, 28.9, 30.0, 31.0, 32.1, 33.0, 34.2, 35.5, 36.3       # 76-91 K
    ])
    return CubicSpline(x_points, y_points)

def t68_from_PSU54(TPSU):
    """
    Calculates the IPTS-68 temperature from the PSU-54 scale.
    Data sourced from R.E. Bedford et al., Metrologia 5, 47-49 (1969).
    Uses iterative cubic spline interpolation to find T68 from TPSU.
    
    Units: 
    - TPSU: K (PSU-54 scale)
    - Output: K (IPTS-68)
    """
    # Range check based on PSU-54 values at the H2 triple point (13.8010 K ) 
    # and the upper limit of Table 2 (approx 91 K ).
    if TPSU < 13.8010 or TPSU > 91:
        raise ValueError("PSU-54 temperature must be between 13.8010 and 91 K.")

    # Initialize Cubic Spline interpolation
    cs = get_PSU54_spline()

    # Iterative solver: T68 = TPSU + Delta_T(T68)
    T_old = TPSU
    while True:
        diff = cs(T_old)
        T_new = TPSU + diff * 1e-3   
        if abs(T_new - T_old) < 1e-12:
            return T_new
        else:
            T_old = T_new

@lru_cache(maxsize=1)
def get_48star_spline():
    """
    Spline for converting IPTS* extension to NBS-55,
    as defined in Table 1 of D.R. Lovejoy, Nature 197, 353 (1963).
    """
    x_points = np.array([
        -222, -220, -218, -216, -214, -212, -210, -208, -206, -204,
        -202, -200, -198, -196, -194, -192, -190, -188, -186, -183
    ])
    # y points: IPTS-48* minus NBS-55 (mK)
    y_points = np.array([
        2273, 1845, 1496, 1209, 973, 780, 623, 495, 390, 304,
        232, 171.8, 122.7, 83.4, 52.9, 30.5, 15.2, 5.7, 1.4, 0.
    ])
    return CubicSpline(x_points, y_points)

def t55_from_48star(T48):
    """
    Calculates the NBS-55 temperature from the IPTS-48* scale extension
    as defined in Table 1 of D.R. Lovejoy, Nature 197, 353 (1963).
    Have to extend to slightly higher T compared to the original table
    to get to the O2 BP that was the bottom of ITS-48.
    
    Units: 
    - T48: K (extended IPTS-48 scale)
    - Output: K (NBS-55)
    """
    # Note that this should only get called for T48 above -222 degC 
    TC48 = T48 - 273.15
# If above range of table but still below O2 boiling point of 90.18 K (on ITS-90),
# assume difference remains zero like it is at 90.15 K in the table.
    if -183 <= TC48 <= -182.97:
        return T48
    elif -222 <= TC48 < -183:
        # Initialize Cubic Spline interpolation
        cs = get_48star_spline()
        diff = cs(TC48)
        return T48 - diff * 1e-3
    else:
        raise ValueError("IPTS48* temperature must be between -182.97 and -222 degC.")

@lru_cache(maxsize=1)
def get_Michels_spline():
    """
    Spline for converting Michels T scale to IPTS-48,
    according to Fig. 1 in Bedford & Prins, Physica 77, 121-125 (1974).
    Points digitized by Google Gemini.
    """
# t_48 (Celsius) from -190 to -1, plus the water triple point at 0.01
    x_array = np.array([
    -190, -189, -188, -187, -186, -185, -184, -183, -182, -181, -180, -179, -178, -177, -176,
    -175, -174, -173, -172, -171, -170, -169, -168, -167, -166, -165, -164, -163, -162, -161,
    -160, -159, -158, -157, -156, -155, -154, -153, -152, -151, -150, -149, -148, -147, -146,
    -145, -144, -143, -142, -141, -140, -139, -138, -137, -136, -135, -134, -133, -132, -131,
    -130, -129, -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116,
    -115, -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102, -101,
    -100, -99, -98, -97, -96, -95, -94, -93, -92, -91, -90, -89, -88, -87, -86, -85, -84, -83,
    -82, -81, -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65,
    -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47,
    -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29,
    -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11,
    -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0.01
    ])

# t_M - t_48 (millikelvins) rounded to 0.01
    y_array = np.array([
    -5.0, -6.39, -7.68, -8.9, -10.03, -11.08, -12.06, -12.97, -13.82, -14.6, -15.32, -15.98, -16.59, -17.15, -17.67, 
    -18.14, -18.58, -18.98, -19.35, -19.69, -20.0, -20.29, -20.57, -20.82, -21.05, -21.26, -21.45, -21.62, -21.77, -21.9, 
    -22.02, -22.11, -22.18, -22.24, -22.28, -22.3, -22.3, -22.29, -22.25, -22.2, -22.14, -22.05, -21.95, -21.83, -21.69, 
    -21.53, -21.36, -21.17, -20.96, -20.74, -20.5, -20.24, -19.97, -19.67, -19.35, -19.01, -18.65, -18.25, -17.83, -17.39, 
    -16.91, -16.4, -15.85, -15.27, -14.65, -14.0, -13.31, -12.58, -11.81, -11.02, -10.2, -9.36, -8.5, -7.62, -6.73, 
    -5.83, -4.93, -4.02, -3.11, -2.21, -1.32, -0.44, 0.43, 1.28, 2.12, 2.94, 3.74, 4.53, 5.31, 6.07, 
    6.81, 7.54, 8.26, 8.96, 9.65, 10.32, 10.98, 11.63, 12.27, 12.89, 13.5, 14.1, 14.68, 15.26, 15.82, 
    16.37, 16.91, 17.44, 17.96, 18.47, 18.97, 19.46, 19.94, 20.42, 20.88, 21.34, 21.78, 22.22, 22.66, 23.08, 
    23.5, 23.91, 24.32, 24.71, 25.1, 25.48, 25.84, 26.2, 26.54, 26.87, 27.18, 27.48, 27.76, 28.03, 28.27, 
    28.5, 28.71, 28.9, 29.06, 29.2, 29.31, 29.4, 29.46, 29.48, 29.48, 29.44, 29.36, 29.25, 29.1, 28.91, 
    28.68, 28.41, 28.11, 27.76, 27.39, 26.97, 26.52, 26.04, 25.52, 24.97, 24.38, 23.77, 23.12, 22.44, 21.74, 
    21.0, 20.24, 19.45, 18.64, 17.8, 16.95, 16.09, 15.21, 14.32, 13.42, 12.51, 11.61, 10.7, 9.79, 8.89, 
    8.0, 7.12, 6.24, 5.39, 4.55, 3.73, 2.93, 2.15, 1.41, 0.69, 0.0
    ])
    return CubicSpline(x_array, y_array)

def t48_from_Michels(TMic):
    """
    Calculates the IPTS-48 (or IPTS48*) temperature from the values on the Michels T scale,
    used at the van der Waals Laboratory in Amsterdam, 
    according to Fig. 1 in Bedford & Prins, Physica 77, 121-125 (1974).
    Scale assumed to be IPTS-48 above 0.01 degC.
     
    Units: 
    - TMic: K (Michels lab scale)
    - Output: K (IPTS-48, or IPTS-48* below O2 boiling point)
    """
    # Convert to Celsius for interpolation
    TC = TMic - 273.15
    # In principle, since the x-axis is T48 instead of the input T, I should iterate to get the correct
    # difference. I will take a shortcut and just iterate once since the difference is small.
    if TC < -190.005:
        raise ValueError("Michels temperature must be at least 83.145 K.")
    elif TC < 0.01:
        cs = get_Michels_spline()
        diff = cs(TC)
        TC2 = TC - diff * 1e-3
        diff2 = cs(TC2)
        return TMic - diff2 * 1e-3
    else:
        return TMic

def main():
    """
    Command-line interface for converting temperature from specified scale to ITS-90.
    Arguments:
    ----------
    -T or -t TEMP (required unless -i is specified)
        TEMP is temperature on the input scale (float)
    -S or -s scale (required) 
        scale is a string indicating the input scale ('IPTS-68', '48', etc.)
    -i input (required if -T/-t is not given)
        input is a string containing a filename with input temperatures, one per line
    -o output (only allowed if -i is given)
        output is a string containing a filename to print output temperatures, one per line
    """
    import sys
    import os
    import argparse
    parser = argparse.ArgumentParser(
        description="Command-line interface for ITS-90 temperature conversions."
    )
    
    # Temperature flags (-t or -T)
    parser.add_argument(
        '-t', '-T',
        type=float,
        dest='temp',
        help="Input temperature (Tin) to convert."
    )
    
    # Scale flags (-s or -S)
    parser.add_argument(
        '-s', '-S',
        type=str,
        dest='scale',
        required=True,
        help="Input scale (e.g., 'IPTS-68', 'PTB-2006')."
    )
    
    # Input file flag (-i)
    parser.add_argument(
        '-i',
        type=str,
        dest='input_file',
        help="Filename containing input values of Tin (one per line)."
    )
    
    # Output file flag (-o)
    parser.add_argument(
        '-o',
        type=str,
        dest='output_file',
        help="Filename to save the converted temperatures (only used with -i)."
    )
    
    args = parser.parse_args()
    
    # --- Constraint Validation ---
    
    # 1. Must use either -t/-T or -i, but not both
    if args.temp is not None and args.input_file is not None:
        print("Error: You cannot use both a single temperature (-t/-T) and an input file (-i).", file=sys.stderr)
        sys.exit(1)
        
    if args.temp is None and args.input_file is None:
        print("Error: You must specify either a temperature (-t/-T) or an input file (-i).", file=sys.stderr)
        sys.exit(1)
        
    # 2. -o can only be used if -i is specified
    if args.output_file is not None and args.input_file is None:
        print("Error: The -o flag can only be used when an input file (-i) is specified.", file=sys.stderr)
        sys.exit(1)
        
    # --- Execution Logic ---
    
    # Case A: Single temperature conversion
    if args.temp is not None:
        try:
            converted_val = t90(args.temp, args.scale)
            print(f"T90 = {converted_val:.7f} K")
        except Exception as e:
            # Print the Error to the screen 
            print(f"{e}", file=sys.stderr)
            sys.exit(1)
            
    # Case B: Bulk conversion from input file
    elif args.input_file is not None:
        # Check if the input file exists
        if not os.path.isfile(args.input_file):
            print(f"Error: The input file '{args.input_file}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        temperatures = []
        results = []
        
        # Read temperatures from file
        try:
            with open(args.input_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue  # skip empty lines
                    try:
                        temperatures.append((line_num, float(stripped)))
                    except ValueError:
                        print(f"Warning: Line {line_num} '{stripped}' is not a valid number. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"Error reading file '{args.input_file}': {e}", file=sys.stderr)
            sys.exit(1)
            
        # Run conversion for each read value
        for line_num, tin in temperatures:
            try:
                converted_val = t90(tin, args.scale)
                results.append(f"{converted_val:.7f}")
            except Exception as e:
                # Print the Error to the screen as requested and write ERROR to file line
                print(f"Error on line {line_num} (Tin={tin}): {e}", file=sys.stderr)
                results.append("ERROR")
                
        # Direct outputs based on the presence of the -o flag
        if args.output_file is not None:
            try:
                with open(args.output_file, 'w') as f:
                    for res in results:
                        f.write(res + '\n')
            except Exception as e:
                print(f"Error writing to output file '{args.output_file}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Print results directly to screen if -o is not provided
            for (_, tin), res in zip(temperatures, results):
                print(f"Tin: {tin} -> T90: {res}")
   

if __name__ == "__main__":
    main()