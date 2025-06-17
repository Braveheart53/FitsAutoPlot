# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:03:26 2021

@author: V448789
"""

#!/home/copper/anaconda3/envs/woot/bin python

# This is a large class for operating a CopperMountain Technologies VNA
# THe commands try to make the operations for intuitive than the raw SCPI commands
# Folling the class statement, a full 1-port calibration is calculated
# from the measurement of Open, Short and Load standards
# Written by Brian Walker, Engineering Manager, Copper Mountain Technologies

import visa
import math
import numpy as np

class S2VNA:
    
    def __init__(self):
        
        rm = visa.ResourceManager()
        #Connect to a Socket on the local machine at 5025
        #Use the IP address of a remote machine to connect to it instead
        try:
            self.resource = rm.open_resource('TCPIP0::127.0.0.1::5025::SOCKET')
        except:
            print("Failure to connect to VNA!")
            print("Check network settings")
        #The VNA ends each line with this. Reads will time out without this
        self.resource.read_termination='\n'
        #Set a really long timeout period for slow sweeps
        #FIXME make this even longer to be safe
        self.resource.timeout = 10000
        
    def start(self, start): # Set the Start Frequency
        self.resource.write('SENS1:FREQ:STAR ' + str(start) + ' HZ\n')
        
    def ifbw(self, ifbw): # Set the IF Bandwidth
        self.resource.write('SENS1:BWID '+str(ifbw)+'\n')

    def stop(self, stop):   # Set the Stop Frequency
        self.resource.write('SENS1:FREQ:STOP ' + str(stop) + ' HZ\n')
        
    def center(self, center):   # Set the Center Frequency
        self.resource.write('SENS1:FREQ:CENT ' + str(center) + ' HZ\n')
        
    def span(self, span):   # Set the Span
        self.resource.write('SENS1:FREQ:SPAN ' + str(span) + ' HZ\n')

    def num_points(self, num_points):   # Set the number of points
        self.resource.write('SENS1:SWE:POIN ' + str(num_points) + '\n')
        
    def num_traces(self, num_traces):   # Set the number of traces
        self.resource.write('CALC1:PAR:COUN ' + str(num_traces) + '\n')
        
    def meas(self, trace, meas):    # Setup trace measurements
        self.resource.write('CALC1:PAR'+str(trace)+':DEF ' + str(meas) + '\n')
        # meas can be S11, S12, S21, S22, A, B, R1, R2

    def getfreq(self):  # Get the frequency data
        self.resource.write('TRIG:SOUR BUS\n')
        self.resource.write('TRIG:SEQ:SING\n')
        self.resource.query('*OPC?\n')

        freq = self.resource.query('SENS1:FREQ:DATA?\n')
        freq = freq.split(',')
        freq = [float(f) for f in freq]
        
        return freq
    
    def getmeas(self, trace): # Single trigger, get measurement, return float list
        self.resource.write('TRIG:SOUR BUS\n')
        self.resource.write('TRIG:SEQ:SING\n')
        self.resource.query('*OPC?\n')
        # SDAT always returns Real-Imag format regardless of display format
        # Wheras FDAT returns data in displayed format
        s = self.resource.query('CALC1:TRAC'+str(trace)+':DATA:SDAT?\n') # Get RI data as string
        s = s.split(',')
        s = [float(m) for m in s]
        
        return s
    
    def tracform(self, trace, form):   # Set the trace format
        # form can be MLOG, PHAS, GDEL, SLIN, SLOG, SCOM, SMIT, SADM, PLIN, PLOG, POL, MLIN, SWR, REAL, IMAG, UPH
        self.resource.write('CALC1:TRAC'+str(trace)+':FORM '+str(form)+'\n')
              
    def kill(self):   # Kill the analyzer process
        self.resource.write('SYST:TERM\n')   
        
    def store(self,file):   # Store the full analyzer state
        #self.resource.write('MMEM:STOR:STYP CDST\n')  # Save everything
        self.resource.write('MMEM:STOR:CDST'+str(file)+'\n')  # Save everything

# Procedure for rotation of cartesion coordinates by radian angle
def rotate(x,y,phi):
  xp = x*math.cos(-phi)-y*math.sin(-phi)
  yp = x*math.sin(-phi)+y*math.cos(-phi)
  return xp,yp  

# Create Procedures to Calculate Reflection Coeff of the Open and Short
# For the CMT S911 cal kit, 50 ohm load is assumed to be perfect
def calcshortRho(freq):
  L0 = 2.0765e-12  # H
  L1 = -108.54e-24  # H/Hz
  L2 = 2.1705e-33  # H/Hz^2
  L3 = -0.01e-42  # H/Hz^3
  Offs = 31.785e-12	# pS
  ind = L0+L1*freq+L2*freq**2+L3*freq**3
  XL = 2*math.pi*freq*ind*1j
  Rho = (XL-50)/(XL+50)
  phi = 2*math.pi*Offs*freq
  x = Rho.real
  y = Rho.imag
  xp,yp = rotate(x,y,phi)
  Rho = complex(xp,yp)
  return Rho
  
def calcopenRho(freq):
  C0 = 49.433e-15  # F
  C1 = -310.13e-27  # F/Hz
  C2 = 23.168e-36  # F/Hz^2
  C3 = -0.15966e-45  # F/Hz^3
  Offs = 29.243e-12	# pS
  cap = C0+C1*freq+C2*freq**2+C3*freq**3
  XC = 1/(2*math.pi*freq*cap*1j)
  Rho = (XC-50)/(XC+50)
  phi = 2*math.pi*Offs*freq
  x = Rho.real
  y = Rho.imag
  xp,yp = rotate(x,y,phi)
  Rho = complex(xp,yp)
  return Rho

# Now create the matrices full of Open, Short, Load "Actuals"
  
numPoints = 801

startFreq = 9e3   # 9 kHz start freq
stopFreq = 6e9   # 6 GHz stop freq

loadRho = np.zeros((numPoints),dtype=complex)
shortRho = np.zeros((numPoints),dtype=complex)
openRho = np.zeros((numPoints),dtype=complex)

for i in range(0,numPoints):
    freq = startFreq+(stopFreq-startFreq)*i/(numPoints-1)
    shortRho[i] = calcshortRho(freq)
    openRho[i] = calcopenRho(freq)

# Now measure the cal kit to get "Measured" values
    
mloadRho = np.zeros((numPoints),dtype=complex)
mshortRho = np.zeros((numPoints),dtype=complex)
mopenRho = np.zeros((numPoints),dtype=complex)
    
# Fire up the VNA and set the start and stop and whatnot
CMT = S2VNA()
CMT.start(startFreq)
CMT.stop(stopFreq)
CMT.num_traces(1)
CMT.meas(1,'S11')
CMT.tracform(1,'MLOG')
CMT.num_points(numPoints)

# Now Measure Open, Short, Load

input('Please attach the Open and pres <Enter>')

S11 = CMT.getmeas(1)
S11r = S11[::2] # Rip out the Real part
S11i = S11[1::2] # Rip out the Imag part
for i in range(0,numPoints):
    mopenRho[i] = complex(S11r[i],S11i[i])

input('Please attach the Short and pres <Enter>')

S11 = CMT.getmeas(1)
S11r = S11[::2] # Rip out the Real part
S11i = S11[1::2] # Rip out the Imag part
for i in range(0,numPoints):
    mshortRho[i] = complex(S11r[i],S11i[i])
    
input('Please attach the Load and pres <Enter>')

S11 = CMT.getmeas(1)
S11r = S11[::2] # Rip out the Real part
S11i = S11[1::2] # Rip out the Imag part
for i in range(0,numPoints):
    mloadRho[i] = complex(S11r[i],S11i[i])

# Now we have to form the Matrices and Solve for Directivity, Reflection Tracking
# and Source Match, D, R and S

# Create the matrices
C = np.zeros((3,3),dtype=complex)
CH = np.zeros((3,3),dtype=complex)
V = np.zeros((3,1),dtype=complex)


# Form the result matrices Directivity, Reflection tracking and Source Match
E = np.zeros((3,1),dtype=complex)

D = np.zeros((numPoints),dtype=complex)
R = np.zeros((numPoints),dtype=complex)
S = np.zeros((numPoints),dtype=complex)

# Fill them with actual and measured values and do the calculation

for i in range(0,numPoints):
    C[0,0] = openRho[i]
    C[0,1] = 1
    C[0,2] = openRho[i]*mopenRho[i]
    C[1,0] = shortRho[i]
    C[1,1] = 1
    C[1,2] = shortRho[i]*mshortRho[i]
    C[2,0] = loadRho[i]
    C[2,1] = 1
    C[2,2] = loadRho[i]*mloadRho[i]
    V[0] = mopenRho[i]
    V[1] = mshortRho[i]
    V[2] = mloadRho[i]
    
    CT = np.transpose(C)
    CH = np.conjugate(CT) #CH is the Hermetian of C
    E1 = CH.dot(C)
    E2 = np.linalg.inv(E1)
    E3 = CH.dot(V)
    E = E2.dot(E3)
    
    D[i] = E[1]
    S[i] = E[2]
    R[i] = E[0]+E[1]*E[2]
    
# We now have the correction factors for the full 1-port Calibration 
# Lets make a measurement
 
dutMeas = np.zeros((numPoints),dtype=complex)
dutCorr = np.zeros((numPoints),dtype=complex)
mdutCorr = np.zeros((numPoints))     
    
input('Please attach a DUT and pres <Enter>')

# Measure a DUT, apply the correction and format to Log mag

S11 = CMT.getmeas(1)
S11r = S11[::2] # Rip out the Real part
S11i = S11[1::2] # Rip out the Imag part


for i in range(0,numPoints):
    dutMeas[i] = complex(S11r[i],S11i[i])  
    dutCorr[i] = (dutMeas[i]-D[i])/(R[i]+S[i]*(dutMeas[i]-D[i]))
    mdutCorr[i] = 20*math.log10(abs(dutCorr[i]))

            
         
   



