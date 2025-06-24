# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:30:29 2025

@author: wwallace
"""

import pyvisa
rm = pyvisa.ResourceManager()
print(rm.list_resources())
# ('ASRL1::INSTR', 'ASRL2::INSTR', 'GPIB0::12::INSTR')
#inst = rm.open_resource('TCPIP0::192.33.116.187::inst0::INSTR')
#print(inst.query("*IDN?"))