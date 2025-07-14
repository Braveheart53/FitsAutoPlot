# Make this into a plugin!

dataName = '10MHz_clk_only_12dBattenInline_1HzRBW_100HzSpan_801pts_freq'
dataNow = GetData(dataName)[0]
offset = mean(dataNow) - 10e6
dataMin = -ptp(dataNow)/2 + offset
dataMax = ptp(dataNow)/2 + offset

newRange = linspace(
    dataMin, dataMax, 801        
    )

rangeName = dataName + '_meanCentered'
numSteps = 801
SetDataRange(rangeName, numSteps, (dataMin, dataMax))