#!/usr/bin/perl

use strict;
use warnings;

foreach $_ (@ARGV) {
  my $oldfile = $_;
  s/grapefruit(-\d+\.png)/gfruit$1/g;
  rename($oldfile, $_);
}
