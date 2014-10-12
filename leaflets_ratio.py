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
parser = argparse.ArgumentParser(prog='ff_contacts_sizes', usage='', add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter, description=\
'''
************************************************
v''' + version_nb + '''
author: Jean Helie (jean.helie@bioch.ox.ac.uk)
git: https://github.com/jhelie/ff_contacts_sizes
DOI: 
************************************************

[ DESCRIPTION ]

This script identities the size of the TM clusters each flip-flopping lipids has been
in contact with.

The identified protein clusters are considered to be transmembrane only if the closest
lipid headgroup neighbours to the cluster particles are all within the same leaflet.

A file listing the flip-flopping lipids must be supplied with the --flipflops option.
Each line of this file should follow the format:

 -> 'resname,resid,starting_leaflet,z_bead'

where starting_leaflet is either 'upper' or 'lower' - e.g. 'POPC,145,lower,PO4'. The
'z_bead' particle is used to track the position of the lipid.


[ REQUIREMENTS ]

The following python modules are needed :
 - MDAnalysis
 - numpy
 - scipy
 - networkX


[ NOTES ]

1. It's a good idea to pre-process the trajectory first and to only output the relevant
   particles (e.g. no water and no cholesterol).

2. Identification of the bilayer leaflets can be controlled via two options.
   (a) beads
    By default, the particles taken into account to define leaflet depend on the
    forcefield (which can be set via the --forcefield option) and are as follows:
    -> Martini: 'name PO4 or name PO3 or name B1A'
   
    Note that only lipids which contain one of the beads mentioned in the selection string
    will be taken into account. If you wish to specify your own selection string (e.g. to
    choose different beads or add a bead not in the default list in order to take into
    account a particular lipid specie) you can do so by supplying a file via the --beads
    option. This file should contain a single line that can be passed as the argument
    to MDAnalysis selectAtoms() routine and should not contain any quotation marks, e.g.:
     -> name PO4 or name PO3 or name B1A or name AM1
        
   (b) leaflet finding method
    By default leaflets are identified using the MDAnalysis LeafletFinder routine and the
    the optimum cutoff to identify 2 lipids groups is determined using the optimize_cutoff
    routine.
    This optimisation process can take time in large systems and you can specify your own
    cutoff value to skip this step. For instance to use a 15 Angstrom cutoff value:
     -> '--leaflet 15'
   
    In very large systems (more then ~50,000 phospholipids) LeafletFinder (or rather the
    networkX module that it relies on) can fail. To  avoid this you can choose not to use
    this routine by specifying:
     -> '--leaflet large'
    In this case lipids whose headgroups z value is above the average lipids z value will
    be considered to make up the upper leaflet and those whose headgroups z value is below
    the average will be considered to be in the lower leaflet.
    This means that the bilayer should be as flat as possible in the gro file supplied in
    order to get a meaningful outcome.

3. Proteins are detected automatically but you can specify an input file to define your
   own selection with the --proteins option.
   In this case the supplied file should contain on each line a protein selection string
   that can be passed as the argument of the MDAnalysis selectAtoms() routine - for 
   instance 'bynum 1:344'.

   
[ USAGE ]
	
Option	      Default  	Description                    
-----------------------------------------------------
-f			: structure file [.gro] (required)
-x			: trajectory file [.xtc]
-o			: name of output folder
-b			: beginning time (ns) (the bilayer must exist by then!)
-e			: ending time (ns)	
-t 		1	: process every t-frames

Lipids identification  
-----------------------------------------------------
--flipflops		: input file with flipflopping lipids, see note 4
--beads			: leaflet identification technique, see note 2(a)
--leaflets	optimise: leaflet identification technique, see note 2(b)

Protein clusters identification and contacts
-----------------------------------------------------
--proteins		: protein selection file, (optional, see note 6)
--pp_cutoff 	8	: cutoff distance for protein-protein contact (Angstrom)
--pl_cutoff 	6	: cutoff distance for protein-lipid contact (Angstrom)
 
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

#lipids identification options
parser.add_argument('--beads', nargs=1, dest='beadsfilename', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('--flipflops', nargs=1, dest='selection_file_ff', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('--leaflets', nargs=1, dest='cutoff_leaflet', default=['optimise'], help=argparse.SUPPRESS)

#protein options
parser.add_argument('--algorithm', dest='m_algorithm', choices=['min'], default='min', help=argparse.SUPPRESS)
parser.add_argument('--proteins', nargs=1, dest='selection_file_prot', default=['auto'], help=argparse.SUPPRESS)
parser.add_argument('--pp_cutoff', nargs=1, dest='cutoff_pp', default=[8], type=float, help=argparse.SUPPRESS)
parser.add_argument('--pl_cutoff', nargs=1, dest='cutoff_pl', default=[6], type=float, help=argparse.SUPPRESS)

#other options
parser.add_argument('--version', action='version', version='%(prog)s v' + version_nb, help=argparse.SUPPRESS)
parser.add_argument('-h','--help', action='help', help=argparse.SUPPRESS)

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

#lipids identification options
args.beadsfilename = args.beadsfilename[0]
args.cutoff_leaflet = args.cutoff_leaflet[0]
args.selection_file_ff = args.selection_file_ff[0]
#radial and protein clusters options
args.selection_file_prot = args.selection_file_prot[0]
args.cutoff_pp = args.cutoff_pp[0]
args.cutoff_pl = args.cutoff_pl[0]

#process options
#---------------

global lipids_ff_nb
lipids_ff_nb = 0
	
#leaflet identification
if args.cutoff_leaflet != "large" and args.cutoff_leaflet != "optimise":
	try:
		args.cutoff_leaflet = float(args.cutoff_leaflet)
	except:
		print "Error: the argument of the --leaflets option should be a number or 'large', see note 2"
		sys.exit(1)

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
	import scipy as sp
	import scipy.stats
except:
	print "Error: you need to install the scipy module."
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
if args.selection_file_ff != "no" and not os.path.isfile(args.selection_file_ff):
	print "Error: file " + str(args.selection_file_ff) + " not found."
	sys.exit(1)
if args.selection_file_prot != "auto" and not os.path.isfile(args.selection_file_prot):
	print "Error: file " + str(args.selection_file_prot) + " not found."
	sys.exit(1)
if args.beadsfilename != "no" and not os.path.isfile(args.beadsfilename):
	print "Error: file " + str(args.beadsfilename) + " not found."
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
	args.output_folder = "ff_ctct_size_" + args.xtcfilename[:-4]

if os.path.isdir(args.output_folder):
	print "Error: folder " + str(args.output_folder) + " already exists, choose a different output name via -o."
	sys.exit(1)
else:
	os.mkdir(args.output_folder)
	
	#create log
	#----------
	filename_log=os.getcwd() + '/' + str(args.output_folder) + '/ff_contacts_sizes.log'
	output_log=open(filename_log, 'w')		
	output_log.write("[ff_contacts_sizes v" + str(version_nb) + "]\n")
	output_log.write("\nThis folder and its content were created using the following command:\n\n")
	tmp_log="python ff_contacts_sizes.py"
	for c in sys.argv[1:]:
		tmp_log+=" " + c
	output_log.write(tmp_log + "\n")
	output_log.close()

	#copy input files
	#----------------
	if args.selection_file_ff != "no":
		shutil.copy2(args.selection_file_ff,args.output_folder + "/")	
	if args.selection_file_prot != "no" and args.selection_file_prot != "auto":
		shutil.copy2(args.selection_file_prot,args.output_folder + "/")
	if args.beadsfilename != "no":
		shutil.copy2(args.beadsfilename,args.output_folder + "/")

##########################################################################################
# FUNCTIONS DEFINITIONS
##########################################################################################

#=========================================================================================
# data loading
#=========================================================================================

def set_lipids_beads():

	global leaflet_sele_string

	#set default beads
	leaflet_beads = {}
	leaflet_beads['martini'] = "name PO4 or name PO3 or name B1A"
	leaflet_sele_string = leaflet_beads['martini']

	#use users input
	if args.beadsfilename != "no":
		with open(args.beadsfilename) as f:
			lines = f.readlines()
		if len(lines) > 1:
			print "Error: the file " + str(args.beadsfilename) + " should conly ontain 1 line (" + str(len(lines)) + " found), see note 2(a)."
			sys.exit(1)
		else:
			if lines[0][-1] == "\n":
				lines[0] = lines[0][:-1]
			leaflet_sele_string = lines[0]

	return
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
		if args.t_start > 0:
			for ts in U.trajectory:
				progress = '\r -skipping frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
				sys.stdout.flush()
				sys.stdout.write(progress)
				if ts.time/float(1000) > args.t_start:
					f_start = ts.frame-1
					break
			print ''
		if (nb_frames_xtc - f_start)%args.frames_dt == 0:
			tmp_offset = 0
		else:
			tmp_offset = 1
		frames_to_process = map(lambda f:f_start + args.frames_dt*f, range(0,(nb_frames_xtc - f_start)//args.frames_dt+tmp_offset))
		nb_frames_to_process = len(frames_to_process)
				
	#check the leaflet selection string is valid
	test_beads = U.selectAtoms(leaflet_sele_string)
	if test_beads.numberOfAtoms() == 0:
		print "Error: invalid selection string '" + str(leaflet_sele_string) + "'"
		print "-> no particles selected."
		sys.exit(1)

	return
def identify_ff():
	print "\nReading selection file for flipflopping lipids..."
	
	#declare variables
	global lipids_ff_nb
	global lipids_ff_info
	global lipids_ff_resnames
	global lipids_ff_leaflet
	global lipids_ff_u2l_index
	global lipids_ff_l2u_index
	global lipids_sele_ff
	global lipids_sele_ff_bead
	global lipids_sele_ff_bonds
	global lipids_sele_ff_VMD_string
	global leaflet_sele_string
	lipids_ff_nb = 0
	lipids_ff_info = {}
	lipids_ff_resnames = []
	lipids_ff_leaflet = []
	lipids_ff_u2l_index = []
	lipids_ff_l2u_index = []
	lipids_sele_ff = {}
	lipids_sele_ff_bead = {}
	lipids_sele_ff_bonds = {}
	lipids_sele_ff_VMD_string={}
		
	with open(args.selection_file_ff) as f:
		lines = f.readlines()
	lipids_ff_nb = len(lines)
	print " -found " + str(lipids_ff_nb) + " flipflopping lipids"
	leaflet_sele_string = leaflet_sele_string + " and not ("
	for l_index in range(0,lipids_ff_nb):
		line = lines[l_index]
		if line[-1] == "\n":
			line = line[:-1]
		try:
			line_content = line.split(',')
			if len(line_content) != 6:
				print "Error: wrong format for line " + str(l_index+1) + " in " + str(args.selection_file_ff) + ", see note 4 in bilayer_perturbations --help."
				print " ->", line
				sys.exit(1)
			#read current lipid details
			lip_resname = line_content[0]
			lip_resnum = int(line_content[1])
			lip_leaflet = line_content[2]
			lip_bead = line_content[3]
			lip_tstart = float(line_content[4])
			lip_tend = float(line_content[5])
			lipids_ff_info[l_index] = [lip_resname,lip_resnum,lip_leaflet,lip_bead,lip_tstart,lip_tend]
						
			#update: starting leaflets
			if lip_leaflet not in lipids_ff_leaflet:
				lipids_ff_leaflet.append(lip_leaflet)

			#update: index in directional lists
			if lip_leaflet == "upper":
				lipids_ff_u2l_index.append(l_index)
			elif lip_leaflet == "lower":
				lipids_ff_l2u_index.append(l_index)
			else:
				print "->unknown starting leaflet '" + str(lip_leaflet) + "'."
				sys.exit(1)
			
			#update: resnames
			if lip_resname not in lipids_ff_resnames:
				lipids_ff_resnames.append(lip_resname)
	
			#update: leaflet selection string
			if l_index==0:
				leaflet_sele_string+="(resname " + str(lip_resname) + " and resnum " + str(lip_resnum) + ")"
			else:
				leaflet_sele_string+=" or (resname " + str(lip_resname) + " and resnum " + str(lip_resnum) + ")"

			#create selections
			lipids_sele_ff[l_index] = U.selectAtoms("resname " + str(lip_resname) + " and resnum " + str(lip_resnum) + " and name " + str(lipids_ff_info[l_index][3]))
			lipids_sele_ff_bead[l_index] = lipids_sele_ff[l_index].selectAtoms("name " + str(lip_bead))
			lipids_sele_ff_VMD_string[l_index]="resname " + str(lipids_ff_info[l_index][0]) + " and resid " + str(lipids_ff_info[l_index][1])
			if lipids_sele_ff[l_index].numberOfAtoms() == 0:
				print "Error:"
				print line
				print "-> no such lipid found."
				sys.exit(1)	
		except:
			print "Error: invalid flipflopping lipid selection string on line " + str(l_index+1) + ": '" + line + "'"
			sys.exit(1)
	leaflet_sele_string+=")"		

	return
def identify_proteins():	
	print "\nIdentifying proteins..."
	
	#import modules
	global nx
	import networkx as nx

	#declare variables
	global proteins_nb
	global proteins_sele
	global proteins_sele_string
	global proteins_sele_string_VMD
	global proteins_boundaries
	global proteins_nb_atoms
	global nb_atom_per_protein	
	proteins_nb = 0
	proteins_sele = {}
	proteins_sele_string = {}
	proteins_sele_string_VMD = {}
	proteins_boundaries = {}
	
	#check for protein presence
	if U.selectAtoms("protein").numberOfAtoms() == 0:
		print "Error: no protein detected."
		sys.exit(1)
	
	#case: selection file provided
	if args.selection_file_prot != "auto":
		print " -reading protein selection file..."
		with open(args.selection_file_prot) as f:
			lines = f.readlines()
		proteins_nb=len(lines)
		proteins_sele["all"] = MDAnalysis.core.AtomGroup.AtomGroup([])
		for p_index in range(0,proteins_nb):
			line = lines[p_index]
			if line[-1] == "\n":
				line = line[:-1]
			progress='\r -creating proteins selections: ' + str(p_index+1) + '/' + str(proteins_nb) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)
			try:
				print " p[" + str(p_index) + "]=U.selectAtoms(" + line + ")"
				proteins_sele[p_index] = U.selectAtoms(line[1:-2])
				proteins_sele["all"] += proteins_sele[p_index]
				proteins_boundaries[p_index] = [proteins_sele[p_index].indices()[0] + 1, proteins_sele[p_index].indices()[proteins_sele[p_index].numberOfAtoms()]+1]
				proteins_sele_string[p_index] = "bynum " + str(proteins_boundaries[p_index][0]) + ":" + str(proteins_boundaries[p_index][1])
				proteins_sele_string_VMD[p_index] = "serial " + str(proteins_boundaries[p_index][0]) + " to " + str(proteins_boundaries[p_index][1])
			except:
				print "Error:"
				print line
				print "->invalid selection string."
				sys.exit(1)
		proteins_nb_atoms = proteins_sele["all"].numberOfAtoms()
	
	#case: automatic detection
	else:
		#declare local variables
		proteins_ca_nb = {}
		proteins_ca_nmax = 0
		proteins_ca_group = {}
		proteins_boundaries = {}
	
		#retrieve 1st atom info
		proteins_sele["all"] = U.selectAtoms("protein")
		proteins_nb_atoms = proteins_sele["all"].numberOfAtoms()
		prec_resnum = proteins_sele["all"][0].resnum
		prec_segid = proteins_sele["all"][0].segid
		prec_atnum = proteins_sele["all"][0].number+1
		prev_atnum = proteins_sele["all"][0].number+1						#atom corresponding to the beginning of the current protein
		#browse following atoms
		for a in proteins_sele["all"][1:]:
			delta_res = a.resnum-prec_resnum
			delta_atm = a.number+1-prec_atnum
			if delta_res < 0 or a.segid != prec_segid or delta_atm > 1:
				proteins_boundaries[proteins_nb] = [prev_atnum,prec_atnum]
				proteins_nb += 1
				prev_atnum = a.number + 1
			prec_resnum = a.resnum
			prec_atnum = a.number + 1
			prec_segid = a.segid		
		#add last protein section
		if prev_atnum < proteins_sele["all"][proteins_nb_atoms-1].number:
			proteins_boundaries[proteins_nb] = [prev_atnum,proteins_sele["all"][proteins_nb_atoms-1].number+1]
			proteins_nb += 1
		
		#display results
		print " -protein found:", proteins_nb
		print " -protein boundaries (atom numbers): see protein.sele file"
		#create protein selections and save into a txt file
		filename_sele=os.getcwd() + '/' + str(args.output_folder) + '/proteins.sele'
		output_stat = open(filename_sele, 'w')	
		output_stat.write("#This file was generated by the script bilayer_perturbations v" + str(version_nb) +"\n")
		output_stat.write("#The lines below correspond to MDAnalysis section string, e.g. U.selectAtoms(LINE)\n")
		output_stat.write("\n")	
		for p_index in range(0, proteins_nb):
			progress='\r -creating proteins selections: ' + str(p_index+1) + '/' + str(proteins_nb) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)
			proteins_sele_string[p_index] = "bynum " + str(proteins_boundaries[p_index][0]) + ":" + str(proteins_boundaries[p_index][1])
			proteins_sele_string_VMD[p_index] = "serial " + str(proteins_boundaries[p_index][0]) + " to " + str(proteins_boundaries[p_index][1])
			proteins_sele[p_index] = U.selectAtoms(proteins_sele_string[p_index])
			output_stat.write(proteins_sele_string[p_index] + "\n")
		output_stat.close()

	nb_atom_per_protein = proteins_sele[0].numberOfAtoms()
	print ""

	return
def identify_leaflets():
	print "\nIdentifying leaflets..."
	
	#declare variables
	global leaflet_sele
	global leaflet_sele_atoms
	leaflet_sele = {}
	leaflet_sele_atoms = {}
	for l in ["lower","upper","both"]:
		leaflet_sele[l] = {}
		leaflet_sele_atoms[l] = {}
	
	#check the leaflet selection string is valid
	test_beads = U.selectAtoms(leaflet_sele_string)
	if test_beads.numberOfAtoms() == 0:
		print "Error: invalid selection string '" + str(leaflet_sele_string) + "'"
		print "-> no particles selected."
		sys.exit(1)

	#use LeafletFinder:
	if args.cutoff_leaflet != 'large':
		if args.cutoff_leaflet == 'optimise':
			print " -optimising cutoff..."
			cutoff_value = MDAnalysis.analysis.leaflet.optimize_cutoff(U, leaflet_sele_string)
			L = MDAnalysis.analysis.leaflet.LeafletFinder(U, leaflet_sele_string, cutoff_value[0])
		else:
			L = MDAnalysis.analysis.leaflet.LeafletFinder(U, leaflet_sele_string, args.cutoff_leaflet)
	
		if np.shape(L.groups())[0]<2:
			print "Error: imposssible to identify 2 leaflets."
			sys.exit(1)
		if L.group(0).centerOfGeometry()[2] > L.group(1).centerOfGeometry()[2]:
			leaflet_sele["upper"]["all species"] = L.group(0)
			leaflet_sele["lower"]["all species"] = L.group(1)
		else:
			leaflet_sele["upper"]["all species"] = L.group(1)
			leaflet_sele["lower"]["all species"] = L.group(0)
		leaflet_sele["both"]["all species"] = leaflet_sele["lower"]["all species"] + leaflet_sele["upper"]["all species"]
		if np.shape(L.groups())[0] == 2:
			print " -found 2 leaflets: ", leaflet_sele["upper"]["all species"].numberOfResidues(), '(upper) and ', leaflet_sele["lower"]["all species"].numberOfResidues(), '(lower) lipids'
		else:
			other_lipids=0
			for g in range(2, np.shape(L.groups())[0]):
				other_lipids += L.group(g).numberOfResidues()
			print " -found " + str(np.shape(L.groups())[0]) + " groups: " + str(leaflet_sele["upper"]["all species"].numberOfResidues()) + "(upper), " + str(leaflet_sele["lower"]["all species"].numberOfResidues()) + "(lower) and " + str(other_lipids) + " (others) lipids respectively"
	#use cof:
	else:
		leaflet_sele["both"]["all species"] = U.selectAtoms(leaflet_sele_string)
		tmp_lipids_avg_z = leaflet_sele["both"]["all species"].centerOfGeometry()[2]
		leaflet_sele["upper"]["all species"] = leaflet_sele["both"]["all species"].selectAtoms("prop z > " + str(tmp_lipids_avg_z))
		leaflet_sele["lower"]["all species"] = leaflet_sele["both"]["all species"].selectAtoms("prop z < " + str(tmp_lipids_avg_z))
		print " -found 2 leaflets: ", leaflet_sele["upper"]["all species"].numberOfResidues(), '(upper) and ', leaflet_sele["lower"]["all species"].numberOfResidues(), '(lower) lipids'
	
	#store full selections
	for l in ["lower","upper","both"]:
		leaflet_sele_atoms[l]["all species"] = leaflet_sele[l]["all species"].residues.atoms
	
	return

#=========================================================================================
# data structures
#=========================================================================================

def data_struct_time():													

	global frames_nb
	global frames_time
	frames_nb = np.zeros(nb_frames_to_process)
	frames_time = np.zeros(nb_frames_to_process)

	return
def data_ff_contacts():

	global lipids_ff_contacts_during_nb
	global lipids_ff_contacts_outside_nb
	global lipids_ff_contacts_during_pc
	global lipids_ff_contacts_outside_pc
	lipids_ff_contacts_during_nb = {}
	lipids_ff_contacts_outside_nb = {}
	lipids_ff_contacts_during_pc = {}
	lipids_ff_contacts_outside_pc = {}
	
	for l_index in range(0,lipids_ff_nb):
		lipids_ff_contacts_during_nb[l_index] = np.zeros(proteins_nb)
		lipids_ff_contacts_outside_nb[l_index] = np.zeros(proteins_nb)
		lipids_ff_contacts_during_pc[l_index] = np.zeros(proteins_nb)
		lipids_ff_contacts_outside_pc[l_index] = np.zeros(proteins_nb)
		
	return

#=========================================================================================
# core functions
#=========================================================================================

def get_z_coords(f_index):														
	
	tmp_zu = leaflet_sele["upper"]["all species"].centerOfGeometry()[2]
	tmp_zl = leaflet_sele["lower"]["all species"].centerOfGeometry()[2]
	tmp_zm = tmp_zl + (tmp_zu - tmp_zl)/float(2)
	z_upper[f_index] = tmp_zu - tmp_zm
	z_lower[f_index] = tmp_zl - tmp_zm
	for l in range(0,lipids_ff_nb):	
		z_ff[l][f_index] = lipids_sele_ff_bead[l].centerOfGeometry()[2] - tmp_zm

	return
def get_distances(box_dim):												
	
	#method: use minimum distance between proteins
	#---------------------------------------------
	if args.m_algorithm == "min":
		#pre-process: get protein coordinates
		tmp_proteins_coords = np.zeros((proteins_nb, nb_atom_per_protein, 3))
		for p_index in range(0, proteins_nb):
			tmp_proteins_coords[p_index,:] = proteins_sele[p_index].coordinates()

		#store min distance between each proteins
		dist_matrix = 100000 * np.ones((proteins_nb,proteins_nb))
		for n in range(proteins_nb,1,-1):
			dist_matrix[proteins_nb-n,proteins_nb-n+1:proteins_nb] = map(lambda pp: np.min(MDAnalysis.analysis.distances.distance_array(np.float32(tmp_proteins_coords[proteins_nb-n,:]), np.float32(tmp_proteins_coords[pp,:]), box_dim)), range(proteins_nb-n+1,proteins_nb))
			dist_matrix[proteins_nb-n+1:proteins_nb,proteins_nb-n] = dist_matrix[proteins_nb-n,proteins_nb-n+1:proteins_nb]
											
	#method: use distance between cog
	#--------------------------------
	else:
		tmp_proteins_cogs = np.asarray(map(lambda p_index: calculate_cog(proteins_sele[p_index].coordinates(), box_dim), range(0,proteins_nb)))
		dist_matrix = MDAnalysis.analysis.distances.distance_array(np.float32(tmp_proteins_cogs), np.float32(tmp_proteins_cogs), box_dim)

	return dist_matrix
def calculate_cog(tmp_coords, box_dim):										
	
	#this method allows to take pbc into account when calculcating the center of geometry 
	#see: http://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
	
	cog_coord = np.zeros(3)
	tmp_nb_atoms = np.shape(tmp_coords)[0]
	
	for n in range(0,3):
		tet = tmp_coords[:,n] * 2 * math.pi / float(box_dim[n])
		xsi = np.cos(tet)
		zet = np.sin(tet)
		tet_avg = math.atan2(-np.average(zet),-np.average(xsi)) + math.pi
		cog_coord[n] = tet_avg * box_dim[n] / float(2*math.pi)
	
	return cog_coord
def detect_clusters_connectivity(dist, box_dim):						
	
	#use networkx algorithm
	connected = (dist<args.cutoff_pp)
	network = nx.Graph(connected)
	groups = nx.connected_components(network)
	
	return groups
def identify_ff_contacts(box_dim, f_time):

	global lipids_ff_contacts_during_nb
	global lipids_ff_contacts_outside_nb
	
	#initialise dictionary allowing to retrieve cluster size
	dict_protatoms_2_clustersize = {}
	
	#retrieve coordinates arrays (pre-processing saves time as MDAnalysis functions are quite slow and we need to make such calls a few times)
	tmp_lip_coords = {l: leaflet_sele[l]["all species"].coordinates() for l in ["lower","upper"]}
	
	#identify clusters
	#=================
	clusters = detect_clusters_connectivity(get_distances(box_dim), box_dim)
	
	#process each cluster
	#====================
	c_sele_all = MDAnalysis.core.AtomGroup.AtomGroup([])
	for cluster in clusters:		
		#create selection for current cluster and only process it if it's TM (find closest PO4 particles for each particles of clusters, if all are in the same leaflet then it's surfacic [NB: this is done at the CLUSTER level (the same criteria at the protein level would probably fail)])
		c_sele = MDAnalysis.core.AtomGroup.AtomGroup([])
		for p_index in cluster:
			c_sele += proteins_sele[p_index]
		c_sele_all += c_sele
		tmp_c_sele_coordinates = c_sele.coordinates()
		dist_min_lower = np.min(MDAnalysis.analysis.distances.distance_array(tmp_c_sele_coordinates, tmp_lip_coords["lower"], box_dim), axis = 1)
		dist_min_upper = np.min(MDAnalysis.analysis.distances.distance_array(tmp_c_sele_coordinates, tmp_lip_coords["upper"], box_dim), axis = 1)
		dist = dist_min_upper - dist_min_lower
		#store size of TM cluster
		if np.size(dist[dist>0]) != np.size(dist) and np.size(dist[dist>0]) !=0:
			c_size = np.size(cluster)
			for a in c_sele.atoms:
				dict_protatoms_2_clustersize[a.number] = c_size		 
		#store -1 for non TM cluster
		else:
			for a in c_sele.atoms:
				dict_protatoms_2_clustersize[a.number] = -1
				
	#process each ff lipid
	#=====================
	for l_index in range(0,lipids_ff_nb):
		#detect contacts
		ff_lip_and_prot_TM = lipids_sele_ff[l_index] + c_sele_all
		around_lip_prot_TM = ff_lip_and_prot_TM.selectAtoms("around " + str(args.cutoff_pl) + " (resname " + str(lipids_ff_info[l_index][0]) + " and resid " + str(lipids_ff_info[l_index][1]) + ")")	
		
		#get size of cluster in contact if any
		if around_lip_prot_TM.numberOfAtoms() > 0:			
			tmp_size = dict_protatoms_2_clustersize[around_lip_prot_TM.atoms[0].number]
			tmp_nbct = around_lip_prot_TM.numberOfAtoms()
						
			#store it if TM
			if tmp_size > 0:
				if f_time < lipids_ff_info[l_index][4] or f_time > lipids_ff_info[l_index][5]:
					lipids_ff_contacts_outside_nb[l_index][tmp_size - 1] += tmp_nbct
				else:
					lipids_ff_contacts_during_nb[l_index][tmp_size - 1] += tmp_nbct
						
	return

#=========================================================================================
# outputs
#=========================================================================================

def write_xvg():
	
	for l_index in lipids_ff_u2l_index:
		filename_xvg = os.getcwd() + '/' + str(args.output_folder) + '/u2l_' + str(lipids_ff_info[l_index][0]) + "_" + str(lipids_ff_info[l_index][1]) + '.xvg'
		output_xvg = open(filename_xvg, 'w')
		output_xvg.write("# [ff_contacts_sizes v" + str(version_nb) + "]\n")
		output_xvg.write("# The data is organised as follows:\n")
		output_xvg.write("#  -1st line: distribution (%) of contacts before/after flipflop\n")
		output_xvg.write("#  -2nd line: distribution (%) of contacts during flipflop\n")
		output_xvg.write("#  -3rd line: distribution (nb) of contacts before/after flipflop\n")
		output_xvg.write("#  -4th line: distribution (nb) of contacts during flipflop\n")
		output_xvg.write("@ title \"Evolution of bilayer thickness by lipid specie\"\n")
		output_xvg.write("@ xaxis label \"cluster size\"\n")
		output_xvg.write("@ yaxis label \"% of contacts\"\n")
		output_xvg.write("@ autoscale ONREAD xaxes\n")
		output_xvg.write("@ TYPE XY\n")
		output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
		output_xvg.write("@ legend on\n")
		output_xvg.write("@ legend box on\n")
		output_xvg.write("@ legend loctype view\n")
		output_xvg.write("@ legend 0.98, 0.8\n")
		output_xvg.write("@ legend length 9\n")
		output_xvg.write("@ s0 legend \"1\"\n")
		output_xvg.write("@ s1 legend \"2\"\n")
		output_xvg.write("@ s2 legend \"3\"\n")
		output_xvg.write("@ s3 legend \"4\"\n")
		output_xvg.write("@ s4 legend \"5\"\n")
		output_xvg.write("@ s5 legend \"6\"\n")
		output_xvg.write("@ s6 legend \"7\"\n")
		output_xvg.write("@ s7 legend \"8\"\n")
		output_xvg.write("@ s8 legend \"9\"\n")
		#distribution: %
		#---------------
		#before/after
		results = "0"
		for c in range(0,9):
			results += "	" + str(round(lipids_ff_contacts_outside_pc[l_index][c],2))
		output_xvg.write(results + "\n")
		#during
		results = "1"
		for c in range(0,9):
			results += "	" + str(round(lipids_ff_contacts_during_pc[l_index][c],2))
		output_xvg.write(results + "\n")
		#distribution: nb
		#---------------
		#before/after
		results = "0"
		for c in range(0,9):
			results += "	" + str(round(lipids_ff_contacts_outside_nb[l_index][c],2))
		output_xvg.write(results + "\n")
		#during
		results = "1"
		for c in range(0,9):
			results += "	" + str(round(lipids_ff_contacts_during_nb[l_index][c],2))
		output_xvg.write(results + "\n")
		output_xvg.close()

	for l_index in lipids_ff_l2u_index:		
		filename_xvg = os.getcwd() + '/' + str(args.output_folder) + '/l2u_' + str(lipids_ff_info[l_index][0]) + "_" + str(lipids_ff_info[l_index][1]) + '.xvg'
		output_xvg = open(filename_xvg, 'w')
		output_xvg.write("# [ff_contacts_sizes v" + str(version_nb) + "]\n")
		output_xvg.write("@ title \"Evolution of bilayer thickness by lipid specie\"\n")
		output_xvg.write("#  -1st line: distribution (%) of contacts before/after flipflop\n")
		output_xvg.write("#  -2nd line: distribution (%) of contacts during flipflop\n")
		output_xvg.write("#  -3rd line: distribution (nb) of contacts before/after flipflop\n")
		output_xvg.write("#  -4th line: distribution (nb) of contacts during flipflop\n")
		output_xvg.write("@ xaxis label \"cluster size\"\n")
		output_xvg.write("@ yaxis label \"% of contacts\"\n")
		output_xvg.write("@ autoscale ONREAD xaxes\n")
		output_xvg.write("@ TYPE XY\n")
		output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
		output_xvg.write("@ legend on\n")
		output_xvg.write("@ legend box on\n")
		output_xvg.write("@ legend loctype view\n")
		output_xvg.write("@ legend 0.98, 0.8\n")
		output_xvg.write("@ legend length 9\n")
		output_xvg.write("@ s0 legend \"1\"\n")
		output_xvg.write("@ s1 legend \"2\"\n")
		output_xvg.write("@ s2 legend \"3\"\n")
		output_xvg.write("@ s3 legend \"4\"\n")
		output_xvg.write("@ s4 legend \"5\"\n")
		output_xvg.write("@ s5 legend \"6\"\n")
		output_xvg.write("@ s6 legend \"7\"\n")
		output_xvg.write("@ s7 legend \"8\"\n")
		output_xvg.write("@ s8 legend \"9\"\n")
		#distribution: %
		#---------------
		#before/after
		results = "0"
		for c in range(0,9):
			results += "	" + str(round(lipids_ff_contacts_outside_pc[l_index][c],2))
		output_xvg.write(results + "\n")
		#during
		results = "1"
		for c in range(0,9):
			results += "	" + str(round(lipids_ff_contacts_during_pc[l_index][c],2))
		output_xvg.write(results + "\n")
		#distribution: nb
		#----------------
		#before/after
		results = "0"
		for c in range(0,9):
			results += "	" + str(round(lipids_ff_contacts_outside_nb[l_index][c],2))
		output_xvg.write(results + "\n")
		#during
		results = "1"
		for c in range(0,9):
			results += "	" + str(round(lipids_ff_contacts_during_nb[l_index][c],2))
		output_xvg.write(results + "\n")
		output_xvg.close()

	return

##########################################################################################
# ALGORITHM
##########################################################################################

#=========================================================================================
#process inputs
#=========================================================================================
#data loading
set_lipids_beads()
load_MDA_universe()
if args.selection_file_ff != "no":
	identify_ff()
identify_proteins()
identify_leaflets()

#create data structures
print "\nInitialising data structures..."
data_struct_time()
data_ff_contacts()

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
	box_dim = U.trajectory.ts.dimensions
	
	#process ff lipids
	identify_ff_contacts(box_dim, f_time)
	
print ''

#=========================================================================================
# process data
#=========================================================================================
print "\nCalculating statistics..."
for l_index in range(0,lipids_ff_nb):
	lipids_ff_contacts_during_pc[l_index] = lipids_ff_contacts_during_nb[l_index] *100 / float(np.sum(lipids_ff_contacts_during_nb[l_index]))
	lipids_ff_contacts_outside_pc[l_index] = lipids_ff_contacts_outside_nb[l_index] *100 / float(np.sum(lipids_ff_contacts_outside_nb[l_index]))
		
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
