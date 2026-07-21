# Description

This repository contains Python code for converting temperatures reported on other scales (mostly obsolete historical scales) to the International Temperature Scale of 1990.  
The conversion procedures are documented in the paper: A.H. Harvey, "Conversion of Reported Temperatures from Historical Scales to the ITS-90," _Int. J. Thermophys._, to be submitted (2026).  

The content consists of the main Python file, ITS90.py, and a test file that can be run with pytest, test_ITS90.py.

# File Documentation

## ITS90.py
--------
The top-level routine to be called for temperature conversion.    
The syntax and documentation of parameters is:  

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

The file may also be run in command-line mode, with a syntax like  
python ITS90.py (-flags)  

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


## test_ITS90.py  

Runs checks of all the conversions.  
Tests limits of validity.  
Tests conversions against Tables 4 and 5 of the related paper.  
Syntax: pytest test_ITS90.py  


# Contact

Allan H. Harvey  
Material Measurement Laboratory, Applied Chemicals and Materials Division, Thermophysical Properties of Fluids Group  
allan.harvey@nist.gov  

# Related Material

A.H. Harvey, "Conversion of Reported Temperatures from Historical Scales to the ITS-90," _Int. J. Thermophys._, to be submitted (2026).  
[DOI link to be added after publication]

