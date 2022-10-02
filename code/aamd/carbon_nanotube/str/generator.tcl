package require topotools
package require inorganicbuilder
package require psfgen
package require solvate
package require autoionize
package require nanotube
mol delete all
mol default style CPK
color Display Background white

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

# Hyperparameters
set file_name %s
set l0 %.5f
set w0 %.5f
set ls %.5f
set n %d
set m %d
set ion_type [list %s]
set ion_conc [list %s]
set ion_valence [list %s]

if {0} {
    set file_name test
    set l0 50.0
    set w0 50.0
    set ls 50.0
    set n 5
    set m 5
    set ion_type [list POT]
    set ion_conc [list 1e-1]
    set ion_valence [list 1]
}


set l0_nm [expr $l0 / 10]
set w0_nm [expr $w0 / 10]
set c_charge 0.0
set cc_bond 1.414
set graphene_vec [expr $cc_bond * sqrt(3)]
# set r0 [expr sqrt($n^2 + $m^2 + $m*$n) * 2.49 / 2 / 3.1415926]

# Nanotube
mol delete all
nanotube -l $l0_nm -m $m -n $n -b 1 -a 0 -i 0 -d 0
center top "all"
center top "all"
center top "all"
set all [atomselect top "all"]
$all set type CA
$all set name CA
$all set segname CN
$all set charge $c_charge
set current_file_name $file_name\_cnt
save $current_file_name
set cnt_minmax [measure minmax $all]
set cnt_minz [expr [lindex [lindex $cnt_minmax 0] 2]]
set cnt_maxz [expr [lindex [lindex $cnt_minmax 1] 2] ]
set r0 [expr [lindex [lindex $cnt_minmax 1] 0] + $cc_bond]

# Graphene 01
mol delete all
graphene -lx $w0_nm -ly $w0_nm -type armchair -b 1 -a 0 -i 0 -d 0
set all [atomselect top "all"]
$all set segname GR1
$all set type CA
$all set name CA
$all set charge $c_charge
center top "all"
set grap [atomselect top "all"]
$grap moveby [list 0 0 $cnt_maxz]
set pore [atomselect top "x^2 + y^2 <= $r0^2"]
set movevec [vecinvert [measure center $pore]]
$grap moveby [list [lindex $movevec 0] [lindex $movevec 1] 0]
set pore [atomselect top "x^2 + y^2 <= $r0^2"]
set pore_atoms [$pore get {segname resid name}]
set current_file_name $file_name\_graphene_pore_01
save $current_file_name
mol delete top
resetpsf
readpsf $current_file_name.psf
coordpdb $current_file_name.pdb
foreach atom $pore_atoms { delatom [lindex $atom 0] [lindex $atom 1] [lindex $atom 2] }
writepsf $current_file_name.psf
writepdb $current_file_name.pdb
mol load psf $current_file_name.psf pdb $current_file_name.pdb

# Graphene 02
mol delete all
graphene -lx $w0_nm -ly $w0_nm -type armchair -b 1 -a 0 -i 0 -d 0
set all [atomselect top "all"]
$all set segname GR2
$all set type CA
$all set name CA
$all set charge $c_charge
center top "all"
set grap [atomselect top "all"]
$grap moveby [list 0 0 $cnt_minz]
set pore [atomselect top "x^2 + y^2 <= $r0^2"]
set movevec [vecinvert [measure center $pore]]
$grap moveby [list [lindex $movevec 0] [lindex $movevec 1] 0]
set pore [atomselect top "x^2 + y^2 <= $r0^2"]
set pore_atoms [$pore get {segname resid name}]
set current_file_name $file_name\_graphene_pore_02
save $current_file_name
mol delete top
resetpsf
readpsf $current_file_name.psf
coordpdb $current_file_name.pdb
foreach atom $pore_atoms { delatom [lindex $atom 0] [lindex $atom 1] [lindex $atom 2] }
writepsf $current_file_name.psf
writepdb $current_file_name.pdb

# Merge
mol delete all
resetpsf

readpsf $file_name\_cnt.psf
readpsf $file_name\_graphene_pore_01.psf
readpsf $file_name\_graphene_pore_02.psf

coordpdb $file_name\_cnt.pdb
coordpdb $file_name\_graphene_pore_01.pdb
coordpdb $file_name\_graphene_pore_02.pdb

writepsf $file_name\_combined.psf
writepdb $file_name\_combined.pdb

# Solvation
mol deleta all
set current_file_name $file_name\_combined
refresh $current_file_name
set all [atomselect top "all"]
set minmax [measure minmax $all]
set min [lindex $minmax 0]
set max [lindex $minmax 1]
set min [lreplace $min  2 2 [expr [lindex $min 2] - $ls]]
set max [lreplace $max  2 2 [expr [lindex $max 2] + $ls]]
set minmax {}
set minmax [linsert $minmax 0 $min $max]
echo $minmax
solvate $current_file_name.psf $current_file_name.pdb -o $file_name\_solvate -minmax $minmax

set current_file_name $file_name\_solvate
refresh $current_file_name
set pore_atoms [atomselect top "water and z <= $cnt_maxz and z >= $cnt_minz"]
ec
set pore_atoms [$pore_atoms get {segname resid name}]
mol delete all
resetpsf
readpsf $current_file_name.psf
coordpdb $current_file_name.pdb
foreach atom $pore_atoms { delatom [lindex $atom 0] [lindex $atom 1] [lindex $atom 2] }
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
set current_file_name $file_name\_ionized

# Add constraint
mol delete all
refresh $current_file_name

set all [atomselect top "all"]
$all set beta 0.0
set sel [atomselect top "not water and not ion"]
$sel set beta 1.0
set surf [atomselect top "not water and not ion and numbonds < 3"]
$surf set beta 10.0
set current_file_name $file_name\_constraint
save $current_file_name

exit