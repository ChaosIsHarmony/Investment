'''
RUN $ python3 -m tests.test_data_processor
'''
import utils.model_generation_engine.data_processor as dp



def test_terminate_early():
    prev_valid_losses = [ 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 ]
    assert dp.terminate_early(prev_valid_losses) == True, "Failed should terminate early increasing in terminate_early test."

    prev_valid_losses.reverse()
    assert dp.terminate_early(prev_valid_losses) == False, "Failed should not terminate early decreasing in terminate_early test."

    prev_valid_losses = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]
    assert dp.terminate_early(prev_valid_losses) == True, "Failed should terminate early stagnating in terminate_early test."



def run_data_processor_tests():
    test_terminate_early()
    print("test_terminate_early() tests all passed.")



if __name__ == "__main__":
    run_data_processor_tests()
