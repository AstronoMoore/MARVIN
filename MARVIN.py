import getpass
import requests
import pandas as pd
from collections import OrderedDict
import json
import numpy as np
from lasair import LasairError, lasair_client as lasair
from astropy.time import Time
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import configparser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.patches as patches
from matplotlib.pyplot import cm
from astropy.coordinates import Distance
import sys
import configparser
import getpass
import yaml


def plot_marvin(marvin_results):
    band_color_index = {}
    tns_name = marvin_results.info['objname'].item()
    redshift = marvin_results.info.redshift.item()
    discovery_mjd = Time(
        marvin_results.info['discoverydate'].item(), format='iso', scale='utc').mjd

    lightcurve = marvin_results.data

    ignore_telescope_list = []
    order = ['W2', 'UVW2', 'M2', 'UVM2', 'W1', 'UVW1', 'U', 'u', 'B', 'b', 'g', 'c', 'V',
             'v', 'r', 'w', 'R', 'G', 'clear', 'o', 'i', 'I', 'z', 'y', 'J', 'H', 'K', 'w1', 'w2']
    offsets = {}
    n = 0
    markersize = 9
    min_offset = int(len(lightcurve.band.unique())/4)

    bands = lightcurve.band.unique()

    bands_for_offset = lightcurve[~lightcurve['telescope'].isin(
        ignore_telescope_list)].band.unique()
    min_offset = int(len(bands_for_offset)/4)
    fillstyle = 'full'
    # bands = lightcurve.band.unique()
    for i in order:
        if i in bands_for_offset:
            offsets[i] = min_offset + n
            n -= 0.5

    fontsize = 20

    fig, ax = plt.subplots(dpi=400, figsize=(10, 7))

    mStyles = [".", "*", "d", "X", "D", ">", "s", "8", "p",
               "P", "h", "H", "+", "x", "X", "D", "d", "|", "_"]
    mStyles = mStyles[:len(lightcurve['telescope'].unique())]

    d = {'telescope': lightcurve['telescope'].unique(
    ), 'mstyle': mStyles[:len(lightcurve['telescope'].unique())]}
    markersyles = pd.DataFrame(data=d)

    n_colors = len(lightcurve['band'].unique())
    color_1_def = cm.plasma(np.linspace(0, 0.9, int(round(n_colors+1)/2)))
    color_2_def = cm.viridis(np.linspace(0.45, 0.9, int(round(n_colors+1)/2)))

    color_1 = iter(color_1_def)
    color_2 = iter(color_2_def)

    # ax.set_xlim(59700,60250)
    count = 0
    handles, labels = [], []

    for filter in order:
        if filter in bands:
            if count < (n_colors / 2):
                c = next(color_1)

            if count >= (n_colors / 2):
                c = next(color_2)

            band_color_index[filter] = c
            count += 1

    for filter in order:
        if filter in bands:

            c = band_color_index[filter]

            single_filter = lightcurve[lightcurve['band'] == filter]
            for telescope in lightcurve['telescope'].unique():
                fillstyle = 'full'
                markersyle = markersyles[markersyles['telescope']
                                         == telescope]['mstyle']
                for_plot = single_filter[single_filter['telescope']
                                         == telescope]
                if for_plot.empty:
                    continue

                if telescope in ignore_telescope_list:
                    continue

                offset = offsets[filter]

                if telescope == 'synthetic':
                    fillstyle = 'none'

                ax.errorbar(for_plot['time'].astype(float), for_plot['magnitude'].astype(float) + offsets[filter], for_plot['e_magnitude'].astype(
                    float), color=c, marker=markersyle.item(), linestyle='', markersize=markersize, fillstyle=fillstyle)
                fillstyle = 'full'

            count += 1

    ax.invert_yaxis()

    color_1 = iter(color_1_def)
    color_2 = iter(color_2_def)

    count = 0

    for i in order:
        if i in bands_for_offset:

            c = band_color_index[i]

            offset = offsets[i]
            if offset < 0:
                offset_str = ''
            else:
                offset_str = '+'
            patch = ax.add_patch(patches.Rectangle(
                (0, 0), 0, 0, facecolor=c, label=i+' '+offset_str+str(offsets[i])))
            count += 1

    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)

    # bands
    ax.legend(handles=handle_list, title="Bands", ncol=8, prop={
              'size': 12}, fontsize=10, title_fontsize=10, markerscale=1.7, loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=False,)

    # point syle
    handle_list, label_list = [], []
    for telescope in lightcurve['telescope'].unique():
        if telescope in ignore_telescope_list:
            continue
        markersyle = markersyles[markersyles['telescope']
                                 == telescope]['mstyle']
        markersyle = markersyle.item()
        handle = ax.errorbar([], [], y_err=None, color='grey', marker=markersyle,
                             linestyle='', label=telescope, fillstyle='none')
        handle_list.append(handle)

    fig.legend(handles=handle_list, title="Telescopes", ncol=8, prop={
               'size': 12}, fontsize=10, title_fontsize=12, markerscale=1.7, loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True, shadow=False)
    plt.tight_layout()

    ax.set_xlabel(r'MJD', fontsize=fontsize)
    ax.set_ylabel(r'Apparent magnitude + offset', fontsize=fontsize)

    if redshift != None:

        DL_Mpc = Distance(z=redshift).Mpc

        mu = 5*np.log10(DL_Mpc/1e-5)

        def app_to_abs(app):
            return app - mu

        def abs_to_app(abs):
            return abs + mu

        def mjd_to_rf(x):
            return (x-discovery_mjd)/(1+redshift)

        def rf_to_mjd(x):
            return x*(1+redshift) + discovery_mjd

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.minorticks_on()
    ax.set_xlim(np.nanmin(lightcurve['time']) -
                10, np.nanmax(lightcurve['time'])+10)
    ax.invert_yaxis()
    ax.invert_yaxis()

    if marvin_results.type == None:
        spec_type = 'Uncalssified'
        print('in if')
    else:
        spec_type = marvin_results.type

    plt.title(tns_name + ' ' + str(spec_type))

    plt.savefig(tns_name+'_lc.pdf', bbox_inches='tight', pad_inches=0.02)


def create_config(token, tns_api_key, tns_api_headers):
    config = configparser.ConfigParser()
    config.add_section('Global')

    config.set('Global', 'Lasair_token', lasair_token)
    config.set('Global', 'tns_api_key', tns_api_key)
    config.set('Global', 'tns_api_headers', tns_api_headers)

    # {'User-Agent':'tns_marker{"tns_id":165250,"type": "bot", "name":"MARVIN"}'}

    # Write the configuration to the file
    with open('MARVIN_config.ini', 'w') as config_file:
        config.write(config_file)


def fetch_config():
    config = configparser.ConfigParser()
    config.read('MARVIN_config.ini')
    api_key = config.get('Global', 'api_key')
    token = config.get('Global', 'token')

    return api_key, token

# caching Pan-STARRS - placeholder data from Ken is hosted on my PESSTO google drive - a proper implementation should be implemented
# This export of the data is OLD!!!
# Some objects will no report Pan-STARRS photometry because I do not have their candidate IDs


def PS_to_internal_name(PSNAME):
    url = 'https://drive.google.com/file/d/1K-Im_ZZUskK6-0yi7VuT4EOB0UBngifn/view?usp=share_link'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    ps1 = pd.read_csv(path, delim_whitespace=True)
    rl = 'https://drive.google.com/file/d/13OR14sOm2U-lZMFT_57v8t1usg2OCHUx/view?usp=share_link'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    ps2 = pd.read_csv(path, delim_whitespace=True)
    panstarrs_good_list = pd.concat((ps1, ps2))

    can_ids = panstarrs_good_list[panstarrs_good_list['ps1_designation'] == PSNAME]
    return can_ids['id'].to_list()


def try_pst_forced(canid):
    canid = str(canid)  # converting to string
    ps1_string = "https://star.pst.qub.ac.uk/sne/ps13pi/psdb/lightcurveforced/"
    ps2_string = "https://star.pst.qub.ac.uk/sne/ps23pi/psdb/lightcurveforced/"
    data = []
    # try ps1
    try:
        data_ps1 = pd.read_csv(ps1_string + canid, delim_whitespace=True)
    except Exception as e:
        print(f'No PS1 for {canid}')
        data_ps1 = pd.DataFrame()

    # try ps2
    try:
        data_ps2 = pd.read_csv(ps1_string + canid, delim_whitespace=True)
    except Exception as e:
        print(f'No PS1 for {canid}')
        data_ps2 = pd.DataFrame()
    # Ifs to serve the available Pan-STARRS data to user

    if data_ps1.empty == False and data_ps2.empty == False:
        output_data = pd.concat((data_ps1, data_ps2)).copy()

    if data_ps1.empty == False and data_ps2.empty == True:
        output_data = data_ps1

    if data_ps1.empty == True and data_ps2.empty == False:
        output_data = data_ps2

    if data_ps1.empty == True and data_ps2.empty == True:
        output_data = None

    return output_data


def fetch_ztf(ztf_name):
    lasair_token = "336663f982474a379a934539e0d09860f6cb69cb"
    L = lasair(lasair_token)
    print('Fetching ZTF through LASAIR')
    objectList = [ztf_name]
    response = L.objects(objectList)
    # create an dictionary of lightcurves
    lcsDict = {}
    for obj in response:
        lcsDict[obj['objectId']] = {'candidates': obj['candidates']}

    cols = {1: 'g', 2: 'r'}
    data = pd.DataFrame(lcsDict[obj['objectId']]['candidates'])
    data = data[data['isdiffpos'] == 't']
    data_ouput = data.filter(
        ['mjd', 'fid', 'magpsf', 'sigmapsf', 'diffmaglim'])
    condition1 = data_ouput['fid'] == 1
    condition2 = data_ouput['fid'] == 2
    replacement_value1 = 'g'
    replacement_value2 = 'r'
    data_ouput.loc[condition1, 'fid'] = replacement_value1
    data_ouput.loc[condition2, 'fid'] = replacement_value2
    data_ouput.insert(len(data_ouput.columns), 'telescope', 'ZTF')
    data_ouput = data_ouput.rename(columns={"fid": "band"})
    data_ouput = data_ouput.rename(columns={"magpsf": "magnitude"})
    data_ouput = data_ouput.rename(columns={"sigmapsf": "e_magnitude"})
    data_ouput = data_ouput.rename(columns={"mjd": "time"})
    return data_ouput


def fetch_ztf_cone(ra, dec):
    lasair_token = "336663f982474a379a934539e0d09860f6cb69cb"
    L = lasair(lasair_token)
    print('Fetching ZTF through LASAIR cone search')
    response = L.cone(ra, dec, radius=1.5, requestType='all')
    data_ouput = fetch_ztf(response[0]['object'])
    return data_ouput


def fetch_gaia(gaia_name):
    print('Fetching GAIA')
    gaia_alerts = 'http://gsaweb.ast.cam.ac.uk/alerts/alert/' + \
        str(gaia_name) + '/lightcurve.csv/'
    data_output = pd.read_csv(gaia_alerts, skiprows=1)
    data_output = data_output[data_output['averagemag'] != 'untrusted']

    data_output = data_output.dropna()
    t_gaia = Time(data_output['JD(TCB)'].values.astype(str), format='jd')
    data_output.insert(len(data_output.columns), 'time', value=t_gaia.mjd)
    data_output.insert(len(data_output.columns), 'band', value='G')
    # properly calculate GAIA mags!!!
    data_output.insert(len(data_output.columns), 'e_magnitude', value='0.1')
    data_output.insert(len(data_output.columns), 'telescope', value='Gaia')
    data_output.insert(len(data_output.columns), 'magnitude',
                       value=data_output['averagemag'].astype(float).round(3))
    data_output = data_output.filter(
        ['time', 'band', 'magnitude', 'e_magnitude', 'telescope'])
    return data_output


def fetch_neowise(ra, dec):
    skycoord = SkyCoord(ra, dec, frame='icrs', unit='deg')
    print('Fetching NEOWISE')
    url = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?catalog=neowiser_p1bs_psd&spatial=cone&radius=5&radunits=arcsec&objstr=" + \
        skycoord.ra.to_string(u.hour, alwayssign=True) + '+' + skycoord.dec.to_string(
            u.degree, alwayssign=True) + "&outfmt=1&selcols=ra,dec,mjd,w1mpro,w1sigmpro,w2mpro,w2sigmpro"
    r = requests.get(url)
    table = Table.read(url, format='ascii')
    neowise_master = table.to_pandas()
    neowise_w1 = neowise_master.filter(['mjd', 'w1mpro', 'w1sigmpro'])
    neowise_w1.insert(len(neowise_w1.columns), 'band', 'w1')
    neowise_w1 = neowise_w1.rename(columns={"w1mpro": "magnitude"})
    neowise_w1 = neowise_w1.rename(columns={"w1sigmpro": "e_magnitude"})
    neowise_w2 = neowise_master.filter(['mjd', 'w2mpro', 'w2sigmpro'])
    neowise_w2.insert(len(neowise_w2.columns), 'band', 'w2')
    neowise_w2 = neowise_w2.rename(columns={"w2mpro": "magnitude"})
    neowise_w2 = neowise_w2.rename(columns={"w2sigmpro": "e_magnitude"})
    neowise = pd.concat((neowise_w1, neowise_w2))
    neowise.insert(len(neowise.columns), 'telescope', 'NEOWISE')
    neowise = pd.concat((neowise_w1, neowise_w2))
    neowise.insert(len(neowise.columns), 'telescope', 'NEOWISE')
    neowise = neowise.rename(columns={"mjd": "time"}).dropna()
    return neowise


def fetch_panstarrs_forced(psname):
    appended_data = []
    candid_list = PS_to_internal_name(psname)
    for candidate_id in candid_list:
        new_data = try_pst_forced(candidate_id)
        appended_data.append(new_data)
        if new_data is None:
            print('no forced for this object!!')
        new_data = try_pst_unforced(candidate_id)
        appended_data.append(try_pst_forced(candidate_id))

    appended_data = pd.concat(appended_data)

    return clean_panstarrs(appended_data)


def fetch_panstarrs_unforced(psname):
    appended_data = []
    candid_list = PS_to_internal_name(psname)
    for candidate_id in candid_list:
        new_data = try_pst_unforced(candidate_id)
        appended_data.append(try_pst_forced(candidate_id))

    appended_data = pd.concat(appended_data)
    return clean_panstarrs(appended_data)


def try_pst_unforced(canid):
    canid = str(canid)  # converting to string
    ps1_string = "https://star.pst.qub.ac.uk/sne/ps13pi/psdb/lightcurve/"
    ps2_string = "https://star.pst.qub.ac.uk/sne/ps23pi/psdb/lightcurve/"
    data = []
    # try ps1
    print('trying ps1')
    try:
        data_ps1 = pd.read_csv(ps1_string + canid, delim_whitespace=True)
    except Exception as e:
        print(f'No PS1 unforced for {canid}')
        data_ps1 = pd.DataFrame()

    # try ps2
    try:
        data_ps2 = pd.read_csv(ps1_string + canid, delim_whitespace=True)
    except Exception as e:
        print(f'No PS1 unforced for {canid}')
        data_ps2 = pd.DataFrame()

    if data_ps1.empty == False and data_ps2.empty == False:
        output_data = pd.concat((data_ps1, data_ps2)).copy()

    if data_ps1.empty == False and data_ps2.empty == True:
        output_data = data_ps1

    if data_ps1.empty == True and data_ps2.empty == False:
        output_data = data_ps2

    if data_ps1.empty == True and data_ps2.empty == True:
        output_data = None
    return output_data


def clean_panstarrs(panstarss):
    panstarss.drop_duplicates(subset='#mjd', keep='first', inplace=True)
    panstarss['mjd_floor'] = np.floor(panstarss['#mjd'])
    panstarss = panstarss.replace(to_replace='None', value=np.nan).dropna()
    grouped = panstarss.groupby(['filter', 'mjd_floor'])
    panstarss = panstarss[panstarss['ujy'] >= 3 * panstarss['dujy']]

    panstarss_output = pd.DataFrame(columns=['mjd', 'm', 'dm', 'filter'])
    count = 0
    for group_name, df_group in grouped:
        working_frame = pd.DataFrame(df_group)
        filter = working_frame['filter'].iloc[0]
        mag = working_frame['cal_psf_mag'].astype(float).mean()
        mag_err = working_frame['cal_psf_mag'].astype(float).std()
        if np.isnan(mag_err) == True:
            mag_err = working_frame['psf_inst_mag_sig'].iloc[0]
        panstarss_output.loc[count] = [
            working_frame['#mjd'].astype(float).mean(), mag, mag_err, filter]
        count += 1

    panstarss_output.insert(len(panstarss_output.columns),
                            'telescope', 'Pan-STARRS')
    panstarss_output.filter(['mjd', 'filter', 'm', 'dm', 'telescope'])

    panstarss_output = panstarss_output.rename(columns={"mjd": "time"})
    panstarss_output = panstarss_output.rename(columns={"m": "magnitude"})
    panstarss_output = panstarss_output.rename(columns={"dm": "e_magnitude"})
    panstarss_output = panstarss_output.rename(columns={"filter": "band"})

    return panstarss_output


def check_output_folder(output_folder):
    # Check if folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder '{output_folder}' created.")
    else:
        print('\n')


class TNS_interogate:
    def __init__(self, TNS_name):
        self.TNS_name = TNS_name
        self.info = None
        self.discoverer = None
        self.internal_names = None
        self.type = None
        self.atlas = None
        self.pan_starrs = None
        self.gaia = None
        self.ztf = None

        url = 'https://www.wis-tns.org/api/get/object'  # TNS URL for object search
        tns_api_key = 'cb0fa7b0ddcc1ed5a4f18a82c1e01fb9092985ec'
        json_file = OrderedDict([("objname", self.TNS_name)])
        data = {'api_key': tns_api_key, 'data': json.dumps(json_file)}
        response = requests.post(url, data=data, headers={
                                 'User-Agent': 'tns_marker{"tns_id":165250,"type": "bot", "name":"MARVIN"}'})

        if response.status_code == 200:
            json_data = json.loads(response.text, object_pairs_hook=dict)
            object_TNS_data = json_data['data']['reply']
            tns_object_info = pd.DataFrame(
                [object_TNS_data.values()], columns=object_TNS_data)
            self.info = tns_object_info
            self.discoverer = tns_object_info.discoverer.item()
            self.type = tns_object_info['object_type'][0]['name']
            self.internal_names = tns_object_info['internal_names'].apply(
                lambda x: x.split(','))[:]
            for item in np.array(self.internal_names[0]):
                item = item.replace(" ", "")
                if isinstance(item, str) and item.startswith("ATLAS"):
                    self.atlas = item
                if isinstance(item, str) and item.startswith("PS"):
                    self.pan_starrs = item
                if isinstance(item, str) and item.startswith("Gaia"):
                    self.gaia = item
                if isinstance(item, str) and item.startswith("ZTF"):
                    self.ztf = item
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            response.headers.get('x-rate-limit-limit')


class Heart_of_Gold:

    def __init__(self, TNS_NAME):
        self.pan_starrs_data = None
        self.gaia_data = None
        self.ztf_data = None
        self.ztf_cone_data = None
        self.neowise_data = None
        self.info = None
        self.type = None
        self.data = pd.DataFrame([])

        TNS_object = TNS_interogate(TNS_NAME)
        self.type = TNS_object.type
        self.info = TNS_object.info

        # Getting ZTF
        if TNS_object.ztf is not None:
            self.ztf_data = fetch_ztf(TNS_object.ztf)
            self.data = pd.concat((self.data, self.ztf_data))
        else:
            print('No ZTF data - risking a conesearch')
            try:
                fetch_ztf_cone(TNS_object.info['radeg'].item(
                ), TNS_object.info['decdeg'].item())
                self.ztf_cone_data = fetch_ztf_cone(
                    TNS_object.info['radeg'], TNS_object.info['decdeg'])
                self.data = pd.concat((self.data, self.ztf_cone_data))
            except LasairError as e:
                print(e)

        if TNS_object.pan_starrs is not None:
            self.pan_starrs_data = fetch_panstarrs_forced(
                TNS_object.pan_starrs)
            self.data = pd.concat((self.data, self.pan_starrs_data))

        self.neowise_data = fetch_neowise(
            TNS_object.info['radeg'].item(), TNS_object.info['decdeg'].item())
        self.data = pd.concat((self.data, self.neowise_data))

        if TNS_object.gaia is not None:
            self.gaia_data = fetch_gaia(TNS_object.gaia)
            self.data = pd.concat((self.data, self.gaia_data))


def MARVIN(TNS_Name):
    # Lasair_token = "336663f982474a379a934539e0d09860f6cb69cb"
    # tns_api_key = 'cb0fa7b0ddcc1ed5a4f18a82c1e01fb9092985ec'
    # tns_api_headers = str(
    #     {'User-Agent': 'tns_marker{"tns_id":165250,"type": "bot", "name":"MARVIN"}'})
    marvin_results = Heart_of_Gold(TNS_Name)
    marvin_results.data.to_csv(TNS_Name + '.csv', index=False)
    plot_marvin(marvin_results)


# Function to get login keys from the user

def get_login_keys():
    print("Please enter your login keys:")
    lasair_token = getpass.getpass("lasair_token: ")
    tns_api_key = getpass.getpass("tns_api_key: ")
    tns_api_headers = getpass.getpass("tns_api_headers: ")

    return lasair_token, tns_api_key, tns_api_headers

# Function to save login keys to a configuration file


def save_login_keys(keys):
    config = configparser.ConfigParser()
    config['LoginKeys'] = {
        'lasair_token': keys[0],
        'tns_api_key': keys[1],
        'tns_api_headers': keys[2]
    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

# Function to load login keys from the configuration file


def load_login_keys():
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        tns_api_headers = config['LoginKeys']['tns_api_headers']
        tns_api_headers = {'User-Agent': tns_api_headers}

        print(tns_api_headers)

        return (
            config['LoginKeys']['lasair_token'],
            config['LoginKeys']['tns_api_key'],
            tns_api_headers
        )
    except (configparser.NoSectionError, configparser.NoOptionError, KeyError):
        keys = None
    return keys


if __name__ == "__main__":
    # Check if the config exists

    keys = load_login_keys()

    if keys is None:
        # If the file doesn't exist or keys are not present, get them from the user
        keys = get_login_keys()
        save_login_keys(keys)

    # Set global variables for login keys
    lasair_token, tns_api_key, tns_api_headers = keys
    print(tns_api_headers)

# if __name__ == "__main__":
#     # Check if the config exists
#     try:
#         keys = load_login_keys()
#     except (configparser.NoSectionError, configparser.NoOptionError, FileNotFoundError):
#         # If the file doesn't exist or keys are not present, get them from the user
#         keys = get_login_keys()
#         save_login_keys(keys)

    # Set global variables for login keys
    # lasair_token, tns_api_key, tns_api_headers = keys

    # Now you can use global_key1, global_key2, and global_key3 in your code
    # print("lasair_token:", lasair_token)
    # print("tns_api_key:", tns_api_key)
    # print("tns_api_headers:", tns_api_headers)

    MARVIN(sys.argv[1])
