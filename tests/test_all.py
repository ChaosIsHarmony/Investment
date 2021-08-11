'''
RUN $ python3 -m tests.test_all
'''
from . import test_data_aggregator as tda
from . import test_data_preprocessor as tdpp
from . import test_data_processor as tdp



def run_tests():
        tda.run_data_aggregator_tests()
        tdpp.run_data_preprocessor_tests()
        tdp.run_data_processor_tests()
        # TODO
        # finish all tests on above three
        # tests on other model_generation_engine files
        # tests on:
        #       portfolio_optimizer
        #       risk_adjusted_return_calculator
        #       signal_generator



if __name__ == "__main__":
    run_tests()
