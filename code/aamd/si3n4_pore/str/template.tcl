# template tcl for building a solvated sio2 nanopore
# Argument:
# - name: name of file preffix
# - box_size: number of lattices in 3 direction
# - pore_radius: radius of nanopores

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
set name %s
set box_size {%d %d %d}; # # of lattice
set pore_radius %.5f; # Angstrom
set pore_angle 10.0

# File name
set file_name $name
set final_file_name str

# SiO2 inorganicBuilder
inorganicBuilder::initMaterials
set box [inorganicBuilder::newMaterialBox Si3N4 {0 0 0} $box_size]
set m [mol new]
inorganicBuilder::buildBox $box $file_name

inorganicBuilder::buildSpecificBonds $box {{SI N 1.78}} {true true false} top

center top "all"
set all [atomselect top all]
$all writepsf $file_name.psf
$all writepdb $file_name.pdb

# Cut nanopore
set psf $file_name.psf
set pdb $file_name.pdb

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
writepsf $file_name.psf
writepdb $file_name.pdb

mol load psf $file_name.psf pdb $file_name.pdb

## Change O to OSI
set oxygen [atomselect top "type N"]
$oxygen set type NSI
$oxygen set mass 14.0067
$oxygen set charge -0.575925
set silicon [atomselect top "type SI"]
$silicon set mass 28.0855
$silicon set charge 0.767900
set all [atomselect top all]
$all writepsf $file_name.psf
$all writepdb $file_name.pdb

exit