# Testing Philosophy

Quick and dirty testing to ensure units work as required
without proper isolation (time consuming mocking, stubbing etc.)

Please be aware that
this introduces interdependencies between test cases and units
and therefore supposed unit tests may not be constrained to the respective unit
and have an integration-like character instead. 

https://docs.python.org/3/library/unittest.html
Note that there is no clean distinction between test fixture 
and unit test in python's unittest model, ie. setUp and 
tearDown methods can be directly specified for a test case
and then result in a test fixture.
For each individual test method of the TestCase class a unique test
fixture is created (each method run in freshly setUp() env).
Multiple tests are run per test case by defining
them as methods of the respective TestCase subclass.