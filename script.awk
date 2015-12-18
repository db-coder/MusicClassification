#!/bin/awk -f
BEGIN{
	FS="  ";
	err = 1000;
	out1 = 0;
	out2 = 0;
	out3 = 0;
}
{
	#print $1, $2
	if($2 < err){
		err = $2; out1 = $5;
		out2 = $6; out3 = $7;
	}
}
END{
	#print err
	print out1
	print out2
	print out3
}
