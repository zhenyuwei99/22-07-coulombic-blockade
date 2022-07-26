# template tcl for building a solvated sio2 nanopore
# Argument:
# - lattice_diameter: number of lattices in x y dimension
# - lattice_height: number of lattices in z dimension
# - pore_radius: radius of nanopores
# - solvation_box_height: Half height of solvation box
# - ion_concentation: concentation of ions

package require inorganicbuilder
package require solvate
package require autoionize
package require psfgen

# Parameters
set lattice_diameter %d; # # of lattice
set lattice_height %d; # # of lattice
set pore_radius %.2f; # Angstrom
set solvation_box_height %.2f; # Angstrom
set ion_concentration %.3f; # mol/L

# File name
set si3n4_cell_file ./str/cell.txt
set si3n4_file_name ./str/si3n4
set si3n4_pore_file_name ./str/si3n4_pore
set si3n4_solvated_file_name ./str/si3n4_solvated
set si3n4_ionized_file_name ./str/si3n4_ionized
set final_file_name ./str/str

# Si3N4 inorganicBuilder
inorganicBuilder::initMaterials
set box [inorganicBuilder::newMaterialHexagonalBox Si3N4 {0 0 0} $lattice_diameter $lattice_height]
set m [mol new]
inorganicBuilder::buildBox $box $si3n4_file_name
inorganicBuilder::buildSpecificBonds $box {{SI N 1.9}} {true true false} top
## Write pdb and psf
set all [atomselect top all]
$all writepsf $si3n4_file_name.psf
$all writepdb $si3n4_file_name.pdb
## Cut pore
set psf $si3n4_file_name.psf
set pdb $si3n4_file_name.pdb
mol load psf $psf pdb $pdb
set pore [atomselect top "sqrt(x^2 + y^2) < $pore_radius"]
set pore_atoms [$pore get {segname resid name}]
$pore delete
mol delete top
## Use psfgen to delete the atoms.
resetpsf
readpsf $psf
coordpdb $pdb
foreach atom $pore_atoms { delatom [lindex $atom 0] [lindex $atom 1] [lindex $atom 2] }
writepsf $si3n4_pore_file_name.psf
writepdb $si3n4_pore_file_name.pdb
mol load psf $si3n4_pore_file_name.psf pdb $si3n4_pore_file_name.pdb
## Change N to NSI
set oxygen [atomselect top "type N"]
$oxygen set type NSI
$oxygen set mass 15.9994
$oxygen set charge -0.575925
set silicon [atomselect top "type SI"]
$silicon set mass 28.0855
$silicon set charge 0.7679
set all [atomselect top all]
$all writepsf $si3n4_pore_file_name.psf
$all writepdb $si3n4_pore_file_name.pdb

# Solvate
solvate $si3n4_pore_file_name.psf \
    $si3n4_pore_file_name.pdb \
    -z $solvation_box_height \
    +z $solvation_box_height \
    -o $si3n4_solvated_file_name
mol delete all
mol load psf $si3n4_solvated_file_name.psf pdb $si3n4_solvated_file_name.pdb
## Cut water
set sin [atomselect top "resname SIN"]
set minmax [measure minmax $sin]
set r [expr {0.5*([lindex $minmax 1 1]-[lindex $minmax 0 1])+2.0}]
set sqrt3 [expr {sqrt(3.0)}]
set cutText "(water) and ((abs(y) < 0.5*$r and abs(x) > 0.5*$sqrt3*$r) or (x > $sqrt3*(y+$r) or x < $sqrt3*(y-$r) or x > $sqrt3*($r-y) or x < $sqrt3*(-y-$r)))"
set cutSel [atomselect top $cutText]
set cutAtoms [lsort -unique [$cutSel get {segname resid}]]
## Use psfgen to delete the atoms.
resetpsf
readpsf $si3n4_solvated_file_name.psf
coordpdb $si3n4_solvated_file_name.pdb
foreach atom $cutAtoms { delatom [lindex $atom 0] [lindex $atom 1] }
writepsf  $si3n4_solvated_file_name.psf
writepdb $si3n4_solvated_file_name.pdb
mol delete all
mol load psf $si3n4_solvated_file_name.psf pdb $si3n4_solvated_file_name.pdb

# Ionized
autoionize -psf $si3n4_solvated_file_name.psf \
    -pdb $si3n4_solvated_file_name.pdb \
    -sc $ion_concentration \
    -o $si3n4_ionized_file_name
mol delete all
mol load psf $si3n4_ionized_file_name.psf pdb $si3n4_ionized_file_name.pdb

# Clean cache file
file delete $si3n4_file_name.pdb
file delete $si3n4_file_name.psf
file delete $si3n4_pore_file_name.pdb
file delete $si3n4_pore_file_name.psf
file delete $si3n4_solvated_file_name.pdb
file delete $si3n4_solvated_file_name.psf
file delete $si3n4_solvated_file_name.log

file rename $si3n4_ionized_file_name.pdb $final_file_name.pdb
file rename $si3n4_ionized_file_name.psf $final_file_name.psf

# Restrain and dumping file
mol delete all
mol load psf $final_file_name.psf pdb $final_file_name.pdb
set all [atomselect top all]
$all set beta 0.0
set sel [atomselect top "resname SIN"]
$sel set beta 1.0
set surf [atomselect top "resname SIN and \
((name \"SI.*\" and numbonds<=3) or (name \"N.*\" and numbonds<=2))"]
$surf set beta 10.0
$all writepdb ${final_file_name}_restrain.pdb

mol delete all
mol load psf $final_file_name.psf pdb $final_file_name.pdb
set all [atomselect top all]
$all set beta 0.0
set sel [atomselect top "resname SIN"]
$sel set beta 1.0
$all writepdb ${final_file_name}_langevin.pdb

# Write cell vector
set out [open $si3n4_cell_file w]
foreach v [inorganicBuilder::getCellBasisVectors $box] { puts $out $v }

set minmax [measure minmax $all]
set min [lindex $minmax 0 2]
set max [lindex $minmax 1 2]
puts $out [expr $max - $min]
close $out

exit