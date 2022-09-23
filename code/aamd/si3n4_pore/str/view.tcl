package require pbctools
proc center {molid sel frame} {
    set selprot [atomselect $molid $sel frame $frame]
    set center_prot [measure center $selprot]
    set transfo [transoffset [vecinvert $center_prot]]
    puts $transfo
    $selprot move $transfo
    set new_center_prot [measure center $selprot]
    puts $new_center_prot
}

# Hyperparameters
set pdb_file 05-sample-nvt-ele.dcd
set psf_file r0-5.000A-w0-53.165A-l0-49.334A-ls-40.000A-POT-5.00e-01molPerL-CAL-1.00e-03molPerL.psf


mol load dcd $pdb_file psf $psf_file

# Warp
# First warp (All SIN)
pbc wrap -center com -centersel "(resname SIN)" -compound residue -all
# Second warp (SIN on one side)
set sin_center [measure center [atomselect top "resname SIN"]]
set z_center [lindex $sin_center 2]
pbc wrap -center com -centersel "(resname SIN) and z < $z_center" -compound residue -all
# Third warp (Water in the pore)
for {set i 0} {$i < 5} {incr i} {
    set minmax [measure minmax [atomselect top "resname SIN"]]
    set minz [lindex [lindex $minmax 0] 2]
    set maxz [lindex [lindex $minmax 1] 2]
    set water [atomselect top "water and z < $maxz and z > $minz"]
    pbc wrap -center origin -shiftcenter [measure center $water] -nocompound -all
}

# center
set num_frames [molinfo top get numframes]
for {set i 0} {$i < $num_frames} {incr i} {
    center top "all" $i
}

display resetview
