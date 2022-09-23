# template tcl for building a solvated sio2 nanopore
# Argument:
# - file_name: file name of file preffix
# - r0: radius of nanopores
# - box_size: number of lattices in 3 direction
# - ls: height of solvation box (one side)

package require inorganicbuilder
package require psfgen
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

proc save {file_name} {
    set all [atomselect top all]
    $all writepsf $file_name.psf
    $all writepdb $file_name.pdb
}

proc refresh {file_name} {
    mol delete all
    mol new
    mol load psf $file_name.psf pdb $file_name.pdb
}

# Parameters
set file_name %s
set pore_radius %.3f
set box_size {%d %d %d}
set solvation_height %.3f
set ion_type [list %s]
set ion_conc [list %s]
set ion_valence [list %s]

if {0} {
    set file_name test
    set box_size {10 10 20}
    set pore_radius 15
    set solvation_height 40
    set ion_type [list CES CAL]
    set ion_conc [list 1e-1 1e-4]
    set ion_valence [list 1 2]
}


# Build Si3N4
inorganicBuilder::initMaterials
set current_file_name $file_name\_origin
set box [inorganicBuilder::newMaterialBox Si3N4 {0 0 0} $box_size]
set m [mol new]
inorganicBuilder::buildBox $box $current_file_name
inorganicBuilder::buildSpecificBonds $box {{SI N 1.78}} {true true false} top
center top "all"
save $current_file_name

# Cut nanopore
set psf $current_file_name.psf
set pdb $current_file_name.pdb
set pi [expr {4.0*atan(1.0)}]
set s0 [expr {0.5*$pore_radius}]
set slope [expr {tan($pore_angle*$pi/180.0)}]
mol load psf $psf pdb $pdb
set pore [atomselect top "sqrt(x^2 + y^2) < $pore_radius"]
# set pore [atomselect top "sqrt(x^2 + y^2) < $s0 + $slope*abs(z)"]
set pore_atoms [$pore get {segname resid name}]
$pore delete
mol delete top
resetpsf
readpsf $psf
coordpdb $pdb
foreach atom $pore_atoms { delatom [lindex $atom 0] [lindex $atom 1] [lindex $atom 2] }
set current_file_name $file_name\_pore
writepsf $current_file_name.psf
writepdb $current_file_name.pdb

# Solvation
set current_file_name $file_name\_pore

refresh $current_file_name
set all [atomselect top "all"]
set minmax [measure minmax $all]
set min [lindex $minmax 0]
set max [lindex $minmax 1]
set min [lreplace $min  2 2 [expr [lindex $min 2] - $solvation_height]]
set max [lreplace $max  2 2 [expr [lindex $max 2] + $solvation_height]]
set minmax {}
set minmax [linsert $minmax 0 $min $max]
echo $minmax
solvate $current_file_name.psf $current_file_name.pdb -o $file_name\_solvate -minmax $minmax
set current_file_name $file_name\_solvate

# Cut water
refresh $current_file_name
set psf $current_file_name.psf
set pdb $current_file_name.pdb
set sin [atomselect top "resname SIN"]
set minmax [measure minmax $sin]
set min [lindex $minmax 0]
set minx [lindex $min 0]
set miny [lindex $min 1]
set max [lindex $minmax 1]
set maxx [lindex $max 0]
set maxy [lindex $max 1]
set sqrt3 [expr {sqrt(3.0)}]
set cutText "(all water) and (y > $sqrt3*x + ($miny-$sqrt3*$minx) or y < $sqrt3*x + ($maxy-$sqrt3*$maxx))"
set cutSel [atomselect top $cutText]
set cutAtoms [lsort -unique [$cutSel get {segname resid}]]
resetpsf
readpsf $psf
coordpdb $pdb
foreach atom $cutAtoms { delatom [lindex $atom 0] [lindex $atom 1] }
set current_file_name $file_name\_cut_solvate
writepsf $current_file_name.psf
writepdb $current_file_name.pdb

# Ionize
refresh $current_file_name
resetpsf
set water [atomselect top "name OH2"]
set cla_concentration 0
for {set i 0} {$i < [llength $ion_valence]} {incr i} {
    set cla_concentration [expr $cla_concentration + [lindex $ion_valence $i] * [lindex $ion_conc $i]]
    echo $cla_concentration
}
set total_concentration [expr $cla_concentration * 2]
set num_per_mol [expr [$water num]/(55.523 + $total_concentration)]
set ion_information [list]
set num_cla_ion 0
for {set i 0} {$i < [llength $ion_valence]} {incr i} {
    set num_ion [expr int(ceil($num_per_mol * [lindex $ion_conc $i] + 0.5))]
    set num_cla_ion [expr int($num_cla_ion + [lindex $ion_valence $i] * $num_ion)]
    set ion_information [linsert $ion_information 0 [
        list [list [lindex $ion_type $i]] $num_ion
    ]]
}
set ion_information [concat $ion_information [list [list CLA $num_cla_ion]]]
echo $ion_information
autoionize -psf $current_file_name.psf -pdb $current_file_name.pdb -nions $ion_information -o $file_name\_ionized
center top "all"
set current_file_name $file_name\_ionized

# Add constraint
refresh $current_file_name
set nitrigen [atomselect top "type N"]
$nitrigen set type NSI
$nitrigen set mass 14.0067
$nitrigen set charge -0.575925
set silicon [atomselect top "type SI"]
$silicon set mass 28.0855
$silicon set charge 0.767900

set all [atomselect top all]
$all set beta 0.0
set sel [atomselect top "resname SIN"]
$sel set beta 1.0
set surf [atomselect top "resname SIN and \
((name \"SI.*\" and numbonds<3) or (name \"N.*\" and numbonds<2))"]
$surf set beta 10.0
set current_file_name $file_name\_constraint
save $current_file_name

exit