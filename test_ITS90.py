import pytest
from ITS90 import t90

def test_t90_conversion_1():
    """
    Test that t90(1, 'X') gives correct increments for all scales listed
    in Table 4 of the paper for T = 1 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 1
    tolerance = 5e-7
    Scales_1 = ['EPT-76','PLTS-2000','He58','He62','He24','He29','He32','He37',
                'BS','He48','L55','55E','PTB-2006','X1']
    Diffs_1 = [-0.006,0.281,3.395,3.395,6.175,26.944,26.952,28.979,
               3.026,3.020,4.854,2.104,0.651,-0.006]
    for Scale, Diff in zip(Scales_1, Diffs_1):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance
    
def test_t90_conversion_2():
    """
    Test that t90(2, 'X') gives correct increments for all scales listed
    in Table 4 of the paper for T = 2 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 2
    tolerance = 5e-7
    Scales_2 = ['EPT-76','He58','He62','He24','He29','He32','He37',
                'BS','He48','L55','55E','PTB-2006','XAc','m(III)',
                'XNML','MAS','X1']
    Diffs_2 = [-0.022,4.228,4.047,-89.483,9.802,13.914,-1.322,-0.454,
               -1.587,6.415,3.611,-0.003,-0.032,0.993,0.701,0.548,-0.022]
    for Scale, Diff in zip(Scales_2, Diffs_2):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance
    
def test_t90_conversion_4():
    """
    Test that t90(4, 'X') gives correct increments for all scales listed
    in Table 4 of the paper for T = 4 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 4
    tolerance = 5e-7
    Scales_4 = ['EPT-76','He58','He24','He29','He32','He37',
                'He48','L55','55E','XAc','m(III)',
                'XNML','MAS','NBS 2-20']
    Diffs_4 = [-0.090,7.052,16.680,7.943,7.943,2.546,1.781,4.056,7.226,
               -0.130,0.496,-0.055,-0.350,-2.290]
    for Scale, Diff in zip(Scales_4, Diffs_4):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance
    
def test_t90_conversion_15():
    """
    Test that t90(15, 'X') gives correct increments for all scales listed
    in Table 4 of the paper for T = 15 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 15
    tolerance = 5e-7
    Scales_15 = ['IPTS-68','NBS68','EPT-76','NBS-55','NBS-39','NPL-61','PRMI-54','PSU-54',
                'XAc','m(III)','XNML','MAS','XPRMI',
                'Giauque','PTR','NBS 2-20']
    Diffs_15 = [-3.236,-3.032,-1.260,-1.230,-11.265,-0.448,-40.527,13.248,
               -1.822,-7.020,-3.457,-4.297,-6.097,31.329,-5.408,-0.660]
    for Scale, Diff in zip(Scales_15, Diffs_15):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance
    
def test_t90_conversion_50():
    """
    Test that t90(50, 'X') gives correct increments for all scales listed
    in Table 4 of the paper for T = 50 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 50
    tolerance = 5e-7
    Scales_50 = ['IPTS-68','NBS68','NBS-55','NBS-39','NPL-61','PRMI-54','PSU-54',
                'Giauque','PTR']
    Diffs_50 = [-5.719,-5.619,7.376,-2.620,-10.319,-10.524,15.659,
               -3.682,28.019]
    for Scale, Diff in zip(Scales_50, Diffs_50):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance
    
def test_t90_conversion_80():
    """
    Test that t90(80, 'X') gives correct increments for all scales listed
    in Table 5 of the paper for T = 80 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 80
    tolerance = 5e-7
    Scales_80 = ['IPTS-68','NBS68','ITS-48','NBS-55','NBS-39','NPL-61',
                'PRMI-54','PSU-54','Giauque','PTR','KOL']
    Diffs_80 = [7.893,4.493,-43.558,-1.109,-11.112,11.298,-6.606,32.311,
               39.322,1.444,3.463]
    for Scale, Diff in zip(Scales_80, Diffs_80):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance

def test_t90_conversion_200():
    """
    Test that t90(200, 'X') gives correct increments for all scales listed
    in Table 5 of the paper for T = 200 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 200
    tolerance = 5e-7
    Scales_200 = ['IPTS-68','NBS68','ITS-48','ITS-27','Michels',
                 'Giauque','PTR','KOL']
    Diffs_200 = [12.286,12.386,45.911,45.911,23.769,18.311,-3.536,17.992]
    for Scale, Diff in zip(Scales_200, Diffs_200):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance
        
def test_t90_conversion_500():
    """
    Test that t90(500, 'X') gives correct increments for all scales listed
    in Table 5 of the paper for T = 500 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 500
    tolerance = 5e-7
    Scales_500 = ['IPTS-68','NBS68','ITS-48','ITS-27','Michels',
                 'KOL','GL']
    Diffs_500 = [-40.441,-40.441,12.786,12.786,12.786,5.771,25.933]
    for Scale, Diff in zip(Scales_500, Diffs_500):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance
        
def test_t90_conversion_1000():
    """
    Test that t90(1000, 'X') gives correct increments for all scales listed
    in Table 5 of the paper for T = 1000 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 1000
    tolerance = 5e-7
    Scales_1000 = ['IPTS-68','NBS68','ITS-48','ITS-27','Michels','GL']
    Diffs_1000 = [12.332,12.332,478.242,788.236,478.242,907.228]
    for Scale, Diff in zip(Scales_1000, Diffs_1000):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance
        
def test_t90_conversion_2000():
    """
    Test that t90(2000, 'X') gives correct increments for all scales listed
    in Table 5 of the paper for T = 2000 K.
    Because of rounding of last digit, use tolerance of 5.e-7.
    """
    Tin = 2000
    tolerance = 5e-7
    Scales_2000 = ['IPTS-68','NBS68','ITS-48','ITS-27','Michels','GL']
    Diffs_2000 = [-558.934,-558.934,2093.111,-1894.418,2093.111,12081.533]
    for Scale, Diff in zip(Scales_2000, Diffs_2000):
        expected_diff = 1.e-3*Diff
        result = t90(Tin,Scale)
        actual_diff = result - Tin
        assert abs(actual_diff - expected_diff) <= tolerance

def test_t90_conversion_NPL75():
    """
    Test that t90(T, 'NPL-75') gives correct increments for both T listed
    in Table 4 of the paper.
    Because of rounding of last digit, use tolerance of 5.e-8.
    """
    tolerance = 5e-8
    Scale = 'NPL-75'
    diff4 = 0.0000*1.e-3
    diff15 = -0.0002*1.e-3
    T4 = t90(4,Scale)
    actual_diff4 = T4 - 4
    assert abs(actual_diff4 - diff4) <= tolerance
    T15 = t90(15,Scale)
    actual_diff15 = T15 - 15
    assert abs(actual_diff15 - diff15) <= tolerance             

def test_t90_out_of_range_value_error():
    """
    Test that t90(10, '68') raises a ValueError.
    """
    with pytest.raises(ValueError, match="T68 cannot be below 13.81 K"):
        t90(10.0, '68')

def test_t90_invalid_scale_raises_error():
    """
    Test that t90(100, 'Invalid') raises a ValueError.
    """
    with pytest.raises(ValueError, match="Scale 'Invalid' not recognized"):
        t90(100, 'Invalid')