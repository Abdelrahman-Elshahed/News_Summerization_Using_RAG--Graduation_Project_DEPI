import os
import pandas as pd
from App.RAG_News import XMLParser
def load_data_from_xml_files(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_parser = XMLParser(os.path.join(directory, filename))
            xml_parser.parse_xml()
            all_data.extend(xml_parser.extract_information())
    return all_data

directory = 'Database/News_Categorization_Files_XML'  
data = load_data_from_xml_files(directory)

df = pd.DataFrame(data)

# Check the DataFrame
print(df.head())