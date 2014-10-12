#generic python modules
import argparse
import operator
from operator import itemgetter
import sys, os, shutil
import os.path

##########################################################################################
# RETRIEVE USER INPUTS
##########################################################################################

#=========================================================================================
# create parser
#=========================================================================================
version_nb="0.0.1"
parser = argparse.ArgumentParser(prog='leaflets_ratio', usage='', add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter, description=\
'''
**********************************************
v''' + version_nb + '''
author: Jean Helie (jean.helie@bioch.ox.ac.uk)
git: https://github.com/jhelie/leaflets_ratio
DOI: 
**********************************************

[ DESCRIPTION ]

This script identities the inter and intra leaflets rations of a given residue.


[ REQUIREMENTS ]

The following python modules are needed :
 - MDAnalysis
 - numpy
 - scipy
 

[ NOTES ]

1. It's a good idea to pre-process the trajectory first to trim it (e.g. remove waters).

2. Lipids whose headgroups z value is above the average lipids z value will be considered
   to make up the upper leaflet and those whose headgroups z value is below the average will
   be considered to be in the lower leaflet. This means that the bilayer should remain quite
   flat in order to get a meaningful outcome.

   
[ USAGE ]
	
Option	      Default  	Description                    
-----------------------------------------------------
-f			: structure file [.gro] (required)
-x			: trajectory file [.xtc]
-o			: name of output folder
-b			: beginning time (ns) (the bilayer must exist by then!)
-e			: ending time (ns)	
-t 		1	: process every t-frames
-r [ROH]	: particle used to select residue to track

Other options
-----------------------------------------------------
--version		: show version number and exit
-h, --help		: show this menu and exit
  
''')

#data options
parser.add_argument('-f', nargs=1, dest='grofilename', default=['no'], help=argparse.SUPPRESS, required=True)
parser.add_argument('-x', nargs=1, dest='xtcfilename', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-o', nargs=1, dest='output_folder', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-b', nargs=1, dest='t_start', default=[-1], type=int, help=argparse.SUPPRESS)
parser.add_argument('-e', nargs=1, dest='t_end', default=[10000000000000], type=int, help=argparse.SUPPRESS)
parser.add_argument('-t', nargs=1, dest='frames_dt', default=[1], type=int, help=argparse.SUPPRESS)
parser.add_argument('-r', nargs=1, dest='beadname', default=['ROH'], help=argparse.SUPPRESS)

#=========================================================================================
# store inputs
#=========================================================================================

#parse user inputs
#-----------------
args = parser.parse_args()
#data options
args.grofilename = args.grofilename[0]
args.xtcfilename = args.xtcfilename[0]
args.output_folder = args.output_folder[0]
args.t_start = args.t_start[0]
args.t_end = args.t_end[0]
args.frames_dt = args.frames_dt[0]
args.beadname = args.beadname[0]

#=========================================================================================
# import modules (doing it now otherwise might crash before we can display the help menu!)
#=========================================================================================

#generic science modules
try:
	import math
except:
	print "Error: you need to install the maths module."
	sys.exit(1)
try:
	import numpy as np
except:
	print "Error: you need to install the numpy module."
	sys.exit(1)
try:
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.colors as mcolors
	mcolorconv = mcolors.ColorConverter()
	import matplotlib.cm as cm				#colours library
	import matplotlib.ticker
	from matplotlib.ticker import MaxNLocator
	from matplotlib.font_manager import FontProperties
	fontP=FontProperties()
except:
	print "Error: you need to install the matplotlib module."
	sys.exit(1)
try:
	import pylab as plt
except:
	print "Error: you need to install the pylab module."
	sys.exit(1)

#MDAnalysis module
try:
	import MDAnalysis
	from MDAnalysis import *
	import MDAnalysis.analysis
	import MDAnalysis.analysis.leaflet
	import MDAnalysis.analysis.distances
	#set MDAnalysis to use periodic boundary conditions
	MDAnalysis.core.flags['use_periodic_selections'] = True
	MDAnalysis.core.flags['use_KDTree_routines'] = False
except:
	print "Error: you need to install the MDAnalysis module first. See http://mdanalysis.googlecode.com"
	sys.exit(1)

#=========================================================================================
# sanity check
#=========================================================================================

if not os.path.isfile(args.grofilename):
	print "Error: file " + str(args.grofilename) + " not found."
	sys.exit(1)
if args.t_end < args.t_start:
	print "Error: the starting time (" + str(args.t_start) + "ns) for analysis is later than the ending time (" + str(args.t_end) + "ns)."
	sys.exit(1)

if args.xtcfilename == "no":
	if '-t' in sys.argv:
		print "Error: -t option specified but no xtc file specified."
		sys.exit(1)
	elif '-b' in sys.argv:
		print "Error: -b option specified but no xtc file specified."
		sys.exit(1)
	elif '-e' in sys.argv:
		print "Error: -e option specified but no xtc file specified."
		sys.exit(1)
	elif '--smooth' in sys.argv:
		print "Error: --smooth option specified but no xtc file specified."
		sys.exit(1)
elif not os.path.isfile(args.xtcfilename):
	print "Error: file " + str(args.xtcfilename) + " not found."
	sys.exit(1)

#=========================================================================================
# create folders and log file
#=========================================================================================
if args.output_folder == "no":
	args.output_folder = "lr_" + args.beadname

if os.path.isdir(args.output_folder):
	print "Error: folder " + str(args.output_folder) + " already exists, choose a different output name via -o."
	sys.exit(1)
else:
	os.mkdir(args.output_folder)
	
	#create log
	#----------
	filename_log=os.getcwd() + '/' + str(args.output_folder) + '/leaflets_ratio.log'
	output_log=open(filename_log, 'w')		
	output_log.write("[leaflets_ratio v" + str(version_nb) + "]\n")
	output_log.write("\nThis folder and its content were created using the following command:\n\n")
	tmp_log="python leaflets_ratio.py"
	for c in sys.argv[1:]:
		tmp_log+=" " + c
	output_log.write(tmp_log + "\n")
	output_log.close()

##########################################################################################
# FUNCTIONS DEFINITIONS
##########################################################################################

#=========================================================================================
# data loading
#=========================================================================================

def load_MDA_universe():												#DONE
	
	global U
	global all_atoms
	global nb_atoms
	global nb_frames_xtc
	global frames_to_process
	global frames_to_write
	global nb_frames_to_process
	global f_start
	global radial_bins
	global radial_bin_max
	global radial_radius_max
	f_start = 0
	if args.xtcfilename == "no":
		print "\nLoading file..."
		U = Universe(args.grofilename)
		all_atoms = U.selectAtoms("all")
		nb_atoms = all_atoms.numberOfAtoms()
		nb_frames_xtc = 1
		frames_to_process = [0]
		frames_to_write = [True]
		nb_frames_to_process = 1
	else:
		print "\nLoading trajectory..."
		U = Universe(args.grofilename, args.xtcfilename)
		all_atoms = U.selectAtoms("all")
		nb_atoms = all_atoms.numberOfAtoms()
		nb_frames_xtc = U.trajectory.numframes
		U.trajectory.rewind()
		#sanity check
		if U.trajectory[nb_frames_xtc-1].time/float(1000) < args.t_start:
			print "Error: the trajectory duration (" + str(U.trajectory.time/float(1000)) + "ns) is shorted than the starting stime specified (" + str(args.t_start) + "ns)."
			sys.exit(1)
		if U.trajectory.numframes < args.frames_dt:
			print "Warning: the trajectory contains fewer frames (" + str(nb_frames_xtc) + ") than the frame step specified (" + str(args.frames_dt) + ")."

		#create list of index of frames to process
		if args.t_end != -1:
			f_end = int((args.t_end*1000 - U.trajectory[0].time) / float(U_timestep))
			if f_end < 0:
				print "Error: the starting time specified is before the beginning of the xtc."
				sys.exit(1)
		else:
			f_end = nb_frames_xtc - 1
		if args.t_start != -1:
			f_start = int((args.t_start*1000 - U.trajectory[0].time) / float(U_timestep))
			if f_start > f_end:
				print "Error: the starting time specified is after the end of the xtc."
				sys.exit(1)
		if (f_end - f_start)%args.frames_dt == 0:
			tmp_offset = 0
		else:
			tmp_offset = 1
		frames_to_process = map(lambda f:f_start + args.frames_dt*f, range(0,(f_end - f_start)//args.frames_dt+tmp_offset))
		nb_frames_to_process = len(frames_to_process)
				
	return

#=========================================================================================
# data structures
#=========================================================================================

def data_time():													

	global frames_nb
	global frames_time
	frames_nb = np.zeros(nb_frames_to_process)
	frames_time = np.zeros(nb_frames_to_process)

	return
def data_sele():
	
	#for leaflet identification
	global leaflets
	global lealfets_nb
	leaflets = U.selectAtoms("name PO4 or name PO3 or name AM1 or name ROH")
	leaflets_nb = leaflets.numberOfAtoms()
	
	#beads of interest
	global specie	
	global specie_nb
	specie = U.selectAtoms("name " + str(args.beadname))
	specie_nb = specie.numberOfAtoms()
	
	return
def data_ratios():

	global ratios_inter
	global ratios_intra
	ratios_inter = {k: np.zeros(nb_frames_to_process) for k in ["upper", "lower"]}
	ratios_intra = {k: np.zeros(nb_frames_to_process) for k in ["upper", "lower"]}
			
	return

#=========================================================================================
# core functions
#=========================================================================================

def calculate_ratios(f_index):														
	
	tmp_z_avg = leaflets.centerOfGeometry()[2]
	tmp_zcoord = leaflets.coordinates()[:,2]
	tmp_s_zcoord = specie.coordinates()[:,2]

	#size of leaflets
	tmp_upper_total = len(tmp_zcoord[tmp_zcoord>tmp_z_avg])
	tmp_lower_total = leaflets_nb - tmp_upper_total

	#nb of beads of interest in each leaflet
	tmp_upper_specie = len(tmp_s_zcoord[tmp_s_zcoord>tmp_z_avg])
	tmp_lower_specie = specie_nb - tmp_upper_specie

	#calculate ratios
	ratios_inter["upper"][f_index] = tmp_upper_specie / float(tmp_upper_total) *100
	ratios_inter["lower"][f_index] = tmp_lower_specie / float(tmp_lower_total) *100
	ratios_intra["upper"][f_index] = tmp_upper_specie / float(tmp_upper_specie + tmp_lower_specie) *100
	ratios_intra["lower"][f_index] = 100 - ratios_intra["upper"][f_index]
	
	return

#=========================================================================================
# outputs
#=========================================================================================

def write_xvg():
	
	filename_xvg = os.getcwd() + '/' + str(args.output_folder) + '/lr_' + str(args.beadname) + '.xvg'
	output_xvg = open(filename_xvg, 'w')
	output_xvg.write("# [leaflets_ratio v" + str(version_nb) + "]\n")
	output_xvg.write("@ title \"Evolution of " + str(args.beadname) + " distribution between leaflets\"\n")
	output_xvg.write("@ xaxis label \"time\"\n")
	output_xvg.write("@ yaxis label \"%\"\n")
	output_xvg.write("@ autoscale ONREAD xaxes\n")
	output_xvg.write("@ TYPE XY\n")
	output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
	output_xvg.write("@ legend on\n")
	output_xvg.write("@ legend box on\n")
	output_xvg.write("@ legend loctype view\n")
	output_xvg.write("@ legend 0.98, 0.8\n")
	output_xvg.write("@ legend length 4\n")
	output_xvg.write("@ s0 legend \"inter upper\"\n")
	output_xvg.write("@ s1 legend \"inter lower\"\n")
	output_xvg.write("@ s2 legend \"intra upper\"\n")
	output_xvg.write("@ s3 legend \"intra lower\"\n")
	for f_index in range(0,nb_frames_to_process): 
		results = str(frames_time[f_index]) + "	" + str(round(ratios_inter["upper"][f_index],2)) + "	" + str(round(ratios_inter["lower"][f_index],2)) + "	" + str(round(ratios_intra["upper"][f_index],2)) + "	" + str(round(ratios_intra["lower"][f_index],2))
		output_xvg.write(results + "\n")
	output_xvg.close()

	return

##########################################################################################
# ALGORITHM
##########################################################################################

#=========================================================================================
#process inputs
#=========================================================================================

load_MDA_universe()
data_time()
data_sele()
data_ratios()

#=========================================================================================
# generate data
#=========================================================================================
print "\nCalculating sizes sampled by flip-flopping lipids..."

for f_index in range(0,nb_frames_to_process):
	ts = U.trajectory[frames_to_process[f_index]]
	if ts.time/float(1000) > args.t_end:
		break
	progress = '\r -processing frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '                      '  
	sys.stdout.flush()
	sys.stdout.write(progress)
			
	#frame properties
	f_time = ts.time/float(1000)
	f_nb = ts.frame
	frames_nb[f_index] = f_nb
	frames_time[f_index] = f_time
	
	#calculate ratios
	calculate_ratios(f_index)
	
print ''

	
#=========================================================================================
# produce outputs
#=========================================================================================
print "\nWriting outputs..."
write_xvg()
					
#=========================================================================================
# exit
#=========================================================================================
print "\nFinished successfully! Check output in ./" + args.output_folder + "/"
print ""
sys.exit(0)
