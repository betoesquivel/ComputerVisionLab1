#!/usr/bin/perl

use strict;
use warnings;

foreach $_ (@ARGV) {
  my $oldfile = $_;
  s/gfruit(-\d+\.png)/grapefruit$1/g;
  rename($oldfile, $_);
}
