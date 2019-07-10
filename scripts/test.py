# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:23:30 2019

@author: gaeta
"""
import ctypes, sys
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False
if not is_admin():
    print("is not admin")
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
from music21 import converter

littleMelody = converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#")
littleMelody.show()