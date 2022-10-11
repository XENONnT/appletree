import appletree as apt
from appletree.utils import get_file_path

# Get parameters
par_config_file_name = get_file_path('er_sr0.json')
par_manager = apt.Parameter(par_config_file_name)
par_manager.sample_init()
parameters = par_manager.get_all_parameter()


def test_likelihood():
    """Test Likelihood"""
    config = dict(
        data_file_name = get_file_path('data_Rn220.csv'),
        bins_type = 'equiprob',
        bins_on = ['cs1', 'cs2'],
        bins = [15, 15],
        x_clip = [0, 100],
        y_clip = [2e2, 1e4],
    )
    llh = apt.Likelihood(**config)
    llh.register_component(apt.components.AC, 'rn220_ac', 'AC_Rn220.pkl')
    llh.register_component(apt.components.ERBand, 'rn220_er')
    llh.print_likelihood_summary(short=True)

    parameters['rn220_ac_rate'] = parameters['ac_rate']
    parameters['rn220_er_rate'] = parameters['er_rate']

    key = apt.randgen.get_key()
    llh.get_log_likelihood(key, int(1e6), parameters)
