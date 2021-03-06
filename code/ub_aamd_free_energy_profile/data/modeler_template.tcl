# template tcl for building a solvated sio2 nanopore
# Argument:
# - box_size: number of lattices in 3 direction
# - pore_radius: radius of nanopores
# - solvation_box_height: Half height of solvation box
# - ion_concentation: concentation of ions

package require inorganicbuilder
package require solvate
package require autoionize


# Parameters
set box_size {%d %d %d}; # # of lattice
set pore_radius %.2f; # Angstrom
set solvation_box_height %.2f; # Angstrom
set ion_concentration %.2f; # mol/L

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

set all [atomselect top all]
$all writepsf $sio2_file_name.psf
$all writepdb $sio2_file_name.pdb

# Cut nanopore
set psf $sio2_file_name.psf
set pdb $sio2_file_name.pdb

mol load psf $psf pdb $pdb
set pore [atomselect top "sqrt(x^2 + y^2) < $pore_radius"]
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


# Solvate
solvate $sio2_pore_file_name.psf \
    $sio2_pore_file_name.pdb \
    -z $solvation_box_height \
    +z $solvation_box_height \
    -o $sio2_solvated_file_name

# Ionized
autoionize -psf $sio2_solvated_file_name.psf \
    -pdb $sio2_solvated_file_name.pdb \
    -sc $ion_concentration \
    -o $sio2_ionized_file_name

# Clean cache file
file delete $sio2_file_name.pdb
file delete $sio2_file_name.psf
file delete $sio2_pore_file_name.pdb
file delete $sio2_pore_file_name.psf
file delete $sio2_solvated_file_name.pdb
file delete $sio2_solvated_file_name.psf

file rename $sio2_ionized_file_name.pdb $final_file_name.pdb
file rename $sio2_ionized_file_name.psf $final_file_name.psf

# Restrain and dumping file
mol delete all
mol load psf $final_file_name.psf pdb $final_file_name.pdb
set all [atomselect top all]
$all set beta 0.0
set sel [atomselect top "resname SIO2"]
$sel set beta 1.0
set surf [atomselect top "resname SIO2 and \
((name \"SI.*\" and numbonds<=3) or (name \"O.*\" and numbonds<=1))"]
$surf set beta 10.0
$all writepdb ${final_file_name}_restrain.pdb

exit