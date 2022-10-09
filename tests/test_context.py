import os
import appletree as apt


def test_context():
    par_config_file_name = os.path.join(
        apt.share.PARPATH,
        'apt_sr0_er.json',
    )
    par_config = apt.utils.load_json(par_config_file_name)
    par_config.update({'rn220_er_rate': par_config['er_rate']})
    par_config.update({'rn220_ac_rate': par_config['ac_rate']})
    context = apt.Context(par_config)

    rn220_llh_config = dict(
        # path to the data file
        data_file_name = os.path.join(
            apt.share.DATAPATH,
            'data_XENONnT_Rn220_v8_strax_v1.2.2_straxen_v1.7.1_cutax_v1.9.0.csv',
        ),
        # binning scheme, 'equiprob' or 'meshgrid'
        bins_type = 'equiprob',
        # dimensions where binning is applied on
        bins_on = ['cs1', 'cs2'],
        # number of bins if bins_type == 'equiprob'
        bins = [30, 30],
        # x range, i.e. cS1 range
        x_clip = [0, 100],
        # y range, i.e. cS2 range
        y_clip = [2e2, 1e4],
    )

    # register the likelihood to the posterior
    context.register_likelihood('rn220_llh', rn220_llh_config)
    # register ER component to rn220_llh
    context.register_component('rn220_llh', apt.components.ERBand, 'rn220_er')
    # register AC component to rn220_llh
    context.register_component('rn220_llh', apt.components.AC, 'rn220_ac')

    context.print_context_summary()
    context.fitting(nwalkers=10, iteration=5)
