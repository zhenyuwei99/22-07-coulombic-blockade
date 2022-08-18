# template tcl for building a solvated sio2 nanopore
# Argument:
# - box_size: number of lattices in 3 direction
# - pore_radius: radius of nanopores
# - solvation_box_height: Half height of solvation box
# - ion_concentation: concentation of ions

package require inorganicbuilder
package require solvate
package require autoionize

proc center {molid sel} {
set selprot [atomselect $molid $sel]
set center_prot [measure center $selprot]
set transfo [transoffset [vecinvert $center_prot]]
puts $transfo
$selprot move $transfo
set new_center_prot [measure center $selprot]
puts $new_center_prot
}

# Parameters
set box_size {20 20 35}; # # of lattice
set pore_radius 20; # Angstrom
set pore_angle 10.0

# File name
set sio2_file_name sio2
set sio2_pore_file_name sio2_pore
set sio2_solvated_file_name sio2_solvated
set sio2_ionized_file_name sio2_ionized
set final_file_name str

# SiO2 inorganicBuilder
inorganicBuilder::initMaterials
set box [inorganicBuilder::newMaterialBox SiO2 {0 0 0} $box_size]
set m [mol new]
inorganicBuilder::buildBox $box $sio2_file_name

inorganicBuilder::buildSpecificBonds $box {{SI O 1.61}} {true true false} top

center top "all"
set all [atomselect top all]
$all writepsf $sio2_file_name.psf
$all writepdb $sio2_file_name.pdb

# Cut nanopore
set psf $sio2_file_name.psf
set pdb $sio2_file_name.pdb

set pi [expr {4.0*atan(1.0)}]
set s0 [expr {0.5*$pore_radius}]
set slope [expr {tan($pore_angle*$pi/180.0)}]
mol load psf $psf pdb $pdb
set pore [atomselect top "sqrt(x^2 + y^2) < $pore_radius"]
# set pore [atomselect top "sqrt(x^2 + y^2) < $s0 + $slope*abs(z)"]
set pore_atoms [$pore get {segname resid name}]
$pore delete
mol delete top

## Use psfgen to delete the atoms.
package require psfgen
resetpsf
readpsf $psf
coordpdb $pdb
foreach atom $pore_atoms { delatom [lindex $atom 0] [lindex $atom 1] [lindex $atom 2] }
writepsf $sio2_pore_file_name.psf
writepdb $sio2_pore_file_name.pdb

mol load psf $sio2_pore_file_name.psf pdb $sio2_pore_file_name.pdb

## Change O to OSI
set oxygen [atomselect top "type O"]
$oxygen set type OSI
$oxygen set mass 15.9994
$oxygen set charge -0.45
set silicon [atomselect top "type SI"]
$silicon set mass 28.0855
$silicon set charge 0.9
set all [atomselect top all]
$all writepsf $sio2_pore_file_name.psf
$all writepdb $sio2_pore_file_name.pdb

exit