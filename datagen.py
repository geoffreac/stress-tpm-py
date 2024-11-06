import os
import pandas as pd
from lxml import etree
# os.chdir('./stress-tpm-py/')

from lxml import etree

# def explore_xml(xmlfile):
#     # Parse the XML file
#     tree = etree.parse(xmlfile)
#     root = tree.getroot()
    
#     # Print the structure of the XML tree for inspection
#     print(etree.tostring(root, pretty_print=True).decode())

# # Example usage:
# xmlfile = './data_source/SP-Corporate-2020-11-01/SP-NAME-YY100116090-Discovery--Inc--2020-11-01-OBLIGOR.xml'
# explore_xml(xmlfile)


def datagen(xmlfile):
    tree = etree.parse(xmlfile)
    root = tree.getroot()
    ns = {"r": "http://xbrl.sec.gov/ratings/2015-03-31"}
    fcd = root.xpath('//r:FCD/text()', namespaces=ns)[0] if root.xpath('//r:FCD/text()', namespaces=ns) else None
    obname = root.xpath('//r:OBNAME/text()', namespaces=ns)[0] if root.xpath('//r:OBNAME/text()', namespaces=ns) else None
    lei = root.xpath('//r:LEI/text()', namespaces=ns)[0] if root.xpath('//r:LEI/text()', namespaces=ns) else None
    
    rows = []
    for ord_elem in root.xpath('//r:ORD', namespaces=ns):

        ip = ord_elem.xpath('r:IP/text()', namespaces=ns)[0] if ord_elem.xpath('r:IP/text()', namespaces=ns) else None
        r = ord_elem.xpath('r:R/text()', namespaces=ns)[0] if ord_elem.xpath('r:R/text()', namespaces=ns) else None
        rad = ord_elem.xpath('r:RAD/text()', namespaces=ns)[0] if ord_elem.xpath('r:RAD/text()', namespaces=ns) else None
        rac = ord_elem.xpath('r:RAC/text()', namespaces=ns)[0] if ord_elem.xpath('r:RAC/text()', namespaces=ns) else None
        rol = ord_elem.xpath('r:ROL/text()', namespaces=ns)[0] if ord_elem.xpath('r:ROL/text()', namespaces=ns) else None
        oan = ord_elem.xpath('r:OAN/text()', namespaces=ns)[0] if ord_elem.xpath('r:OAN/text()', namespaces=ns) else None
        rt = ord_elem.xpath('r:RT/text()', namespaces=ns)[0] if ord_elem.xpath('r:RT/text()', namespaces=ns) else None
        rst = ord_elem.xpath('r:RST/text()', namespaces=ns)[0] if ord_elem.xpath('r:RST/text()', namespaces=ns) else None
        
        rows.append({
            "FCD": fcd,
            "OBNAME": obname,
            "LEI": lei,
            "IP": ip,
            "R": r,
            "RAD": rad,
            "RAC": rac,
            "ROL": rol,
            "OAN": oan,
            "RT": rt,
            "RST": rst
        })
    return pd.DataFrame(rows)

def list_xml_files(directory, pattern="-OBLIGOR.xml"):
    return [os.path.join(directory, f) for f in os.listdir(directory) if pattern in f and f.endswith('.xml')]

def process_all_xml_files(directory):
    all_data = []
    selected_files = list_xml_files(directory)
    for xmlfile in selected_files:
        all_data.append(datagen(xmlfile)) 
    return pd.concat(all_data, ignore_index=True)

def load_or_generate_full_data(directory, csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"Generating new data from XML files in {directory}")
        full_data = process_all_xml_files(directory)
        full_data.to_csv(csv_path, index=False)
        return full_data

xml_directory = './data_source/SP-Corporate-2020-11-01'
csv_output_path = './data_source/generated_data/full_data.csv'
full_data = load_or_generate_full_data(xml_directory, csv_output_path)

def clean_rating(data):
    data = data[data['RST'] == 'Local Currency LT']
    
    replacements = {
        "AA-": "AA", "AA+": "AA", 
        "A-": "A", "A+": "A", 
        "BBB-": "BBB", "BBB+": "BBB", 
        "BB-": "BB", "BB+": "BB", 
        "B-": "B", "B+": "B", 
        "CCC-": "CCC", "CCC+": "CCC", 
        "CC": "CCC", "SD": "D"
    }
    data['R'] = data['R'].replace(replacements)
    
    order_levels = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D", "NR"]
    data['R'] = pd.Categorical(data['R'], categories=order_levels, ordered=True)

    col_order = ["OBNAME","LEI","IP", "R", "RAD", "RAC", "ROL", "RST", "RT"]
    data = data[col_order]
    data['RAD'] = pd.to_datetime(data['RAD'])
    return data

def load_or_generate_dt(full_data,dt_path):
    if os.path.exists(dt_path):
        return pd.read_pickle(dt_path)
    else:
        print(f"Generating simplified rating data in {dt_path}")
        dt = clean_rating(full_data)
        dt.to_pickle(dt_path)
        return dt

dt_path ='./data_source/generated_data/dt.pkl'
dt = load_or_generate_dt(full_data,dt_path)

def generate_detailed_history(dt):
    dt['RAD'] = pd.to_datetime(dt['RAD'])
    fulltime = pd.date_range(start=dt['RAD'].min(), end=dt['RAD'].max())
    listdt = dt['OBNAME'].unique()
    
    history_frames = []
    
    for name in listdt:
        temp = dt[dt['OBNAME'] == name].sort_values(by='RAD')
        history = pd.DataFrame({'Corporate': name, 'Rating': None, 'time': fulltime})
        
        for i in range(len(temp)):
            row = temp.iloc[i]
            start_date = row['RAD']
            
            if i == len(temp) - 1:
                end_date = fulltime[-1]
            else:
                next_row = temp.iloc[i + 1]
                end_date = next_row['RAD'] - pd.Timedelta(days=1)
            
            history.loc[(history['time'] >= start_date) & (history['time'] <= end_date), 'Rating'] = row['R']
        
        history_frames.append(history)
    
    return pd.concat(history_frames, ignore_index=True)

def load_or_generate_S_all(dt, S_all_path):
    if os.path.exists(S_all_path):
        print(f"Loading existing detailed history data from {S_all_path}")
        return pd.read_pickle(S_all_path)
    else:
        print(f"Generating detailed history data and saving it to {S_all_path}")
        S_all = generate_detailed_history(dt)
        S_all.to_pickle(S_all_path)
        return S_all

S_all_path = './data_source/generated_data/S_all.pkl'
S_all = load_or_generate_S_all(dt, S_all_path)

import matplotlib.pyplot as plt

def plot_rating_distribution(S_all):

    temp_data = S_all.groupby(['time', 'Rating']).size().reset_index(name='Count')
    temp_data['prop'] = temp_data.groupby('time')['Count'].apply(lambda x: x / x.sum())
    temp_data = temp_data[~temp_data['Rating'].isin([None, 'D', 'NR'])]
    temp_data['total'] = temp_data.groupby('time')['Count'].transform('sum')

    for rating in temp_data['Rating'].unique():
        subset = temp_data[temp_data['Rating'] == rating]
        plt.plot(subset['time'], subset['Count'], label=rating)
    
    # plt.plot(temp_data['time'].unique(), temp_data.groupby('time')['total'].first(), label='Total', linestyle='--', color='black')

    plt.xlabel('Time')
    plt.ylabel('Rating Distribution')
    plt.legend()
    plt.show()


plot_rating_distribution(S_all)
print('stop')


