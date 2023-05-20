import segyio
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#filename = 'small.sgy'
# with segyio.open(filename) as segyfile:

#     # Memory map file for faster reading (especially if file is big...)
#     # segyfile.mmap()

#     # # Print binary header info
#     # print(segyfile.bin)
#     # print(segyfile.bin[segyio.BinField.Traces])

#     # # Read headerword inline for trace 10
#     # print(segyfile.header[10][segyio.TraceField.INLINE_3D])

#     # # Print inline and crossline axis
#     # print(segyfile.xlines)
#     # print(segyfile.ilines)
    
#     for trace in segyfile.trace:
#         # filtered = trace[np.where(trace < 1e-2)]
#         #print(len(trace))
#         print(trace)



def parseSegy(filename):

    
    with segyio.open(filename, ignore_geometry=True) as segyfile:
        segyfile.mmap()

        print(segyfile.bin)
        # file_header = segyfile.text[0]
        # print("------File Header: ------")
        # print(file_header)
        print(segyfile.bin[segyio.BinField.Traces])


        # Extract header word for all traces
        sourceX = segyfile.attributes(segyio.TraceField.SourceX)[:]
        # Scatter plot sources and receivers color-coded on their number
        sourceY = segyfile.attributes(segyio.TraceField.SourceY)[:]
        nsum = segyfile.attributes(segyio.TraceField.NSummedTraces)[:]

        SampleInterval = segyfile.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[:]
        SampleCount = segyfile.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[:]
        OffsetNumber = segyfile.attributes(segyio.TraceField.offset)[:]


        # print(sourceX)
        # print(sourceY)
        # print(nsum)
        print("Sample Interval: " , SampleInterval)
        print("Sample Count: " , SampleCount)
        print("Offset: " , OffsetNumber)
        print("Number of Trace: ", len(OffsetNumber))

        count = 0
        for trace in segyfile.trace:
            count = count + 1
            # filtered = trace[np.where(trace < 1e-2)]
            print(len(trace))
            print(type(trace))
            print(trace)
            if(count == 3):
                break
        #print(count)
        TotalTraceNumber = len(SampleInterval)
        TotalTraceNumber = 2100
        SampleCountNumber = SampleCount[0]
        SampleIntervalNumber = SampleInterval[0]

        xAxis = np.zeros(TotalTraceNumber * SampleCountNumber)
        yAxis = np.zeros(TotalTraceNumber * SampleCountNumber)
        values = np.zeros(TotalTraceNumber * SampleCountNumber)
        TraceMatrix = np.zeros((SampleCountNumber, TotalTraceNumber))
        count = 0
        for trace in segyfile.trace:#16425

            # xAxis = np.append(xAxis, np.full((len(trace)), count))
            # yAxis = np.append(yAxis, np.arange(len(trace)))
            # values = np.append(values, trace)
            i = 0
            while i < len(trace):#2049
                xAxis[SampleCountNumber*count + i] = count + 1
                yAxis[SampleCountNumber*count + i] = (i + 1) * (SampleIntervalNumber/1000)
                values[SampleCountNumber*count + i] = trace[i]
                TraceMatrix[i][count] = trace[i]
                i = i + 1
            
            count = count + 1
            print(count)
            if(count == TotalTraceNumber):
                break
        
        print("Trace matrix shape is: ", TraceMatrix.shape)
        
        return TraceMatrix,SampleIntervalNumber

        plt.scatter(xAxis, yAxis, c=values, cmap='Greys')
        plt.gca().invert_yaxis()
        plt.title(filename)
        plt.xlabel("Offset(TraceNumber)")
        plt.ylabel("TWTT")
        plt.show()

        # plt.figure()
        # plt.scatter(sourceX, sourceY, c=nsum, edgecolor='none')
        # plt.show()

        # groupX = segyfile.attributes(segyio.TraceField.GroupX)[:]
        # groupY = segyfile.attributes(segyio.TraceField.GroupY)[:]
        # nstack = segyfile.attributes(segyio.TraceField.NStackedTraces)[:]
        # plt.scatter(groupX, groupY, c=nstack, edgecolor='none')
        # plt.show()
