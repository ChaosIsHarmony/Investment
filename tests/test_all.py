'''
RUN $ python3 -m tests.test_all
'''
from . import test_data_aggregator as tda
from . import test_data_preprocessor as tdpp
from . import test_data_processor as tdp
from . import test_dataset_methods as tdm
from . import test_signal_generator as tsg
from . import test_risk_adjusted_return_calculator as trarc
from . import test_portfolio_optimizer as tpo



def run_tests():
        tda.run_data_aggregator_tests()
        tdpp.run_data_preprocessor_tests()
        tdp.run_data_processor_tests()
        tdm.run_dataset_methods_tests()
        tsg.run_signal_generator_tests()
        trarc.run_risk_adjusted_return_calculator_tests()
        tpo.run_portfolio_optimzer_tests()


if __name__ == "__main__":
    run_tests()
