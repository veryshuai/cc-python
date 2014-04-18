from nose.tools import *
import sales_tax
 
def setup():
    print "SETUP!"

def teardown():
    print "TEAR DOWN!"

def test_basic():
    sales_tax.__init__
    print "I RAN!"
