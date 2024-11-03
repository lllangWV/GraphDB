from glob import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging


logger = logging.getLogger(__name__)

class Edges:

    def __init__(self, relationship_type, relationship_dir, node_dir, output_format='pandas'):
        
        if output_format not in ['pandas', 'pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        self.relationship_type = relationship_type
        self.relationship_dir = relationship_dir
        self.node_manager =  NodeManager(node_dir=node_dir)  # Store the NodeManager instance
        os.makedirs(self.relationship_dir, exist_ok=True)

        self.output_format = output_format
        self.file_type = 'parquet'
        self.filepath = os.path.join(self.relationship_dir, f'{self.relationship_type}.{self.file_type}')
        self.schema = self.create_schema()

        self.get_dataframe()

    def get_dataframe(self, columns=None, include_cols=True, from_scratch=False, remove_duplicates=True, **kwargs):
        """
        Loads or creates the relationship data based on the specified parameters.

        Parameters
        ----------
        columns : list, optional
            A list of columns to include or exclude from the DataFrame.
        include_cols : bool, optional
            Whether to include or exclude the specified columns (default is True).
        from_scratch : bool, optional
            If True, forces the creation of the relationship data from scratch (default is False).
        remove_duplicates : bool, optional
            Whether to remove duplicate relationships (default is True).

        Returns
        -------
        pd.DataFrame or pyarrow.Table
            The loaded or newly created relationship DataFrame.

        Raises
        ------
        ValueError
            If required fields ('start_node_id' and 'end_node_id') are missing in the relationship DataFrame.
        """

        start_node_type,connection_name,end_node_type=self.relationship_type.split('-')
        start_node_name=f'{start_node_type}-START_ID'
        end_node_name=f'{end_node_type}-END_ID'

       
        if os.path.exists(self.filepath) and not from_scratch:
            logger.info(f"Trying to load {self.relationship_type} relationships from {self.filepath}")
            df = self.load_dataframe(filepath=self.filepath, columns=columns, include_cols=include_cols, **kwargs)
            return df

        logger.info(f"No relationship file found. Attempting to create {self.relationship_type} relationships")
        df = self.create_relationships(**kwargs)  # Subclasses will define this

        # Ensure the 'start_node_id' and 'end_node_id' fields are present
        if start_node_name not in df.columns or end_node_name not in df.columns:
            raise ValueError(f"'{start_node_name}' and '{end_node_name}' fields must be defined for {self.relationship_type} relationships.")
        
        # If 'weight' is not in the dataframe, add it or if remove_duplicates is True, remove duplicates
        if 'weight' not in df.columns:
            if remove_duplicates:
                df=self.remove_duplicate_relationships(df)

        df['TYPE'] = self.relationship_type

        if columns:
            df = df[columns]

        if not self.schema:
            logger.error(f"No schema set for {self.relationship_type} relationships")
            return None

        self.save_dataframe(df, self.filepath)
        return df
    
    def get_property_names(self):
        """
        Retrieves and logs the names of properties (columns) in the relationship file.

        Returns
        -------
        list
            A list of property names in the relationship file.
        """
        properties = Relationships.get_column_names(self.filepath)
        for property in properties:
            logger.info(f"Property: {property}")
        return properties

    def create_relationships(self, **kwargs):
        """
        Abstract method for creating relationships. Must be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            If this method is called from the base class instead of a subclass.
        """
        if self.__class__.__name__ != 'Relationships':
            raise NotImplementedError("Subclasses must implement this method.")
        else:
            pass

    def create_schema(self, **kwargs):
        """
        Abstract method for creating the schema for relationships. Must be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            If this method is called from the base class instead of a subclass.
        """
        if self.__class__.__name__ != 'Relationships':
            raise NotImplementedError("Subclasses must implement this method.")
        else:
            pass

    def load_dataframe(self, filepath, columns=None, include_cols=True, **kwargs):
        """
        Loads a DataFrame from a parquet file.

        Parameters
        ----------
        filepath : str
            The path to the parquet file.
        columns : list, optional
            List of columns to include or exclude when loading the file.
        include_cols : bool, optional
            Whether to include or exclude the specified columns (default is True).

        Returns
        -------
        pd.DataFrame or pyarrow.Table
            The loaded DataFrame or table, depending on the output format.

        Raises
        ------
        Exception
            If an error occurs while loading the file.
        """
        try:
            if self.output_format == 'pandas':
                df = pd.read_parquet(filepath, columns=columns)
            elif self.output_format == 'pyarrow':
                df = pq.read_table(filepath, columns=columns)
            return df
        except Exception as e:
            logger.error(f"Error loading {self.relationship_type} relationships from {filepath}: {e}")
            return None

    def save_dataframe(self, df, filepath):
        """
        Saves a DataFrame to a parquet file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to save.
        filepath : str
            The path where the DataFrame will be saved.

        Raises
        ------
        Exception
            If an error occurs while saving the DataFrame to a parquet file.
        """
        try:
            parquet_table = pa.Table.from_pandas(df, self.schema)
            pq.write_table(parquet_table, filepath)
            logger.info(f"Finished saving {self.relationship_type} relationships to {filepath}")
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")

    def to_neo4j(self, save_dir):
        """
        Converts the relationship data to Neo4j-compatible CSV format.

        Parameters
        ----------
        save_dir : str
            The directory where the Neo4j CSV file will be saved.
        """
        logger.info(f"Converting relationship to Neo4j : {self.filepath}")

        relationship_type=os.path.basename(self.filepath).split('.')[0]
        node_a_type,connection_name,node_b_type=relationship_type.split('-')

        logger.debug(f"Relationship type: {relationship_type}")

        metadata = pq.read_metadata(self.filepath)
        column_types = {}
        neo4j_column_name_mapping={}
        for filed_schema in metadata.schema:
            # Only want top column names
            type=filed_schema.physical_type
            
            field_path=filed_schema.path.split('.')
            name=field_path[0]

            is_list=False
            if len(field_path)>1:
                is_list=field_path[1] == 'list'

            column_types[name] = {}
            column_types[name]['type']=type
            column_types[name]['is_list']=is_list
            
            if type=='BYTE_ARRAY':
               neo4j_type ='string'
            if type=='BOOLEAN':
                neo4j_type='boolean'
            if type=='DOUBLE':
                neo4j_type='float'
            if type=='INT64':
                neo4j_type='int'

            if is_list:
                neo4j_type+='[]'

            column_types[name]['neo4j_type'] = f'{name}:{neo4j_type}'
            column_types[name]['neo4j_name'] = f'{name}:{neo4j_type}'

            neo4j_column_name_mapping[name]=f'{name}:{neo4j_type}'

        neo4j_column_name_mapping['TYPE']=':LABEL'

        neo4j_column_name_mapping[f'{node_a_type}-START_ID']=f':START_ID({node_a_type}-ID)'
        neo4j_column_name_mapping[f'{node_b_type}-END_ID']=f'END_ID({node_a_type}-ID)'

        df=self.load_relationships(filepath=self.filepath)


        df.rename(columns=neo4j_column_name_mapping, inplace=True)

        os.makedirs(save_dir,exist_ok=True)

       
        save_file=os.path.join(save_dir,f'{relationship_type}.csv')

        logger.debug(f"Saving {relationship_type} relationship_path to {save_file}")

        df.to_csv(save_file, index=False)

        logger.info(f"Finished converting relationship to Neo4j : {relationship_type}")

    def validate_nodes(self):
        """
        Validate that the nodes used in the relationships exist in the node manager.
        """
        start_nodes = set(self.get_dataframe()['start_node'].unique())
        end_nodes = set(self.get_dataframe()['end_node'].unique())
        
        existing_nodes = self.node_manager.get_existing_nodes()
        missing_start_nodes = start_nodes - existing_nodes
        missing_end_nodes = end_nodes - existing_nodes

        if missing_start_nodes:
            logger.warning(f"Missing start nodes: {missing_start_nodes}")
        if missing_end_nodes:
            logger.warning(f"Missing end nodes: {missing_end_nodes}")

        return not missing_start_nodes and not missing_end_nodes
    
    @staticmethod
    def get_column_names(filepath):
        metadata = pq.read_metadata(filepath)
        all_columns = []
        for filed_schema in metadata.schema:
            
            # Only want top column names
            max_defintion_level=filed_schema.max_definition_level
            if max_defintion_level!=1:
                continue

            all_columns.append(filed_schema.name)
        return all_columns
    
    @staticmethod
    def remove_duplicate_relationships(df):
        """Expects only two columns with the that represent the id of the nodes

        Parameters
        ----------
        df : pandas.DataFrame
        """
        column_names=list(df.columns)

        df['id_tuple'] = df.apply(lambda x: tuple(sorted([x[column_names[0]], x[column_names[1]]])), axis=1)
        # Group by the sorted tuple and count occurrences
        grouped = df.groupby('id_tuple')
        weights = grouped.size().reset_index(name='weight')
        
        # Drop duplicates based on the id_tuple
        df_weighted = df.drop_duplicates(subset='id_tuple')

        # Merge with weights
        df_weighted = pd.merge(df_weighted, weights, on='id_tuple', how='left')

        # Drop the id_tuple column
        df_weighted = df_weighted.drop(columns='id_tuple')
        return df_weighted
    
