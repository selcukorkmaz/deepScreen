 $base='/Users/selcukkorkmaz//Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/sdf/combined';open(IN,"<$base.pdb");@indata = <IN>;$i=0;
 foreach $line(@indata) {
 if($line =~ /^COMPND/) {++$i;$file="$line.pdb";$file =~ s/COMPND    //g;$file =~ s/[\$#@~!&*()\[\]?;,:^ `\\\/]+//g;$file =~ s/[\r\n]+//g;open(OUT,">/Users/selcukkorkmaz//Documents/Studies/DeepCNNandMLP/exercise2/imageData/pubchem/ligands/$file");next}
 if($line =~ /^ENDMDL/) {next}
 if($line =~ /^ATOM/ || $line =~ /^HETATM/) {print OUT "$line"}
 }

