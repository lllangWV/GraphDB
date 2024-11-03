from glob import glob
import os
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

from matgraphdb import config
# from matgraphdb.utils.chem_utils.coord_geometry import mp_coord_encoding
from matgraphdb.graph_kit.metadata import get_node_schema
from matgraphdb.graph_kit.metadata import NodeTypes

logger = logging.getLogger(__name__)



class Nodes:
    """
    A base class to manage node operations, including creating, loading, and saving nodes as Parquet files, 
    with options to format data as either Pandas or PyArrow DataFrames. Subclasses should implement custom 
    logic for node creation and schema generation.

    Attributes:
    -----------
    node_type : str
        The type of node being created or managed, used in file naming and within the node data.
    node_dir : str
        Directory path where the node files will be stored.
    output_format : str, optional
        Format for loading data, either 'pandas' (default) or 'pyarrow'. Controls the output format when loading the node data.
    file_type : str
        The file type for storing the nodes, which is set to 'parquet'.
    filepath : str
        Full path to the node file, combining the node directory and file type.
    schema : pyarrow.Schema or None
        The schema of the Parquet file. Must be implemented by subclasses.

    Methods:
    --------
    get_dataframe(columns=None, include_cols=True, from_scratch=False, **kwargs):
        Loads or creates a node dataframe. If the file exists, it will load it; otherwise, 
        it will call the create_nodes() method, which must be implemented by subclasses.
    
    get_property_names():
        Returns the names of the columns (properties) found in the Parquet file.
    
    create_nodes(**kwargs):
        Abstract method for creating nodes. This should be implemented by subclasses to define 
        the custom logic for creating nodes.
    
    create_schema(**kwargs):
        Abstract method for creating a schema. This should be implemented by subclasses to define 
        the schema for the Parquet file.
    
    load_dataframe(filepath, columns=None, include_cols=True, **kwargs):
        Loads the nodes from a Parquet file, optionally filtering by columns.
    
    save_dataframe(df, filepath):
        Saves the provided dataframe to a Parquet file at the specified filepath.

    to_neo4j(save_dir):
        Converts the node data to a format suitable for import into Neo4j, saving the resulting CSV file 
        in the specified directory.

    get_column_names(filepath):
        Static method to retrieve the column names from the metadata of a Parquet file.
    """
    def __init__(self, node_type, node_dir, output_format='pandas'):
        """
        Initializes a Nodes object with the given node type, directory, and output format.

        Parameters:
        -----------
        node_type : str
            The type of node to manage.
        node_dir : str
            Directory where node files will be stored.
        output_format : str, optional
            Format for loading data, either 'pandas' (default) or 'pyarrow'. Must be one of these two options.
        
        Raises:
        -------
        ValueError
            If output_format is not 'pandas' or 'pyarrow'.
        """
        if output_format not in ['pandas','pyarrow']:
            raise ValueError("output_format must be either 'pandas' or 'pyarrow'")
        
        self.node_type = node_type
        self.node_dir = node_dir
        os.makedirs(self.node_dir,exist_ok=True)

        self.output_format = output_format
        self.file_type = 'parquet'
        self.filepath =  os.path.join(self.node_dir, f'{self.node_type}.{self.file_type}')
        self.schema = self.create_schema() 

        self.get_dataframe()
        
    def get_dataframe(self, columns=None, include_cols=True, from_scratch=False, **kwargs):
        """
        Loads or creates a node dataframe. If the node file exists and from_scratch is False, 
        the existing file will be loaded. Otherwise, it will call the create_nodes() method to 
        create the nodes and save them.

        Parameters:
        -----------
        columns : list of str, optional
            A list of column names to load. If None, all columns will be loaded.
        include_cols : bool, optional
            If True (default), the specified columns will be included. If False, 
            they will be excluded from the loaded dataframe.
        from_scratch : bool, optional
            If True, forces the creation of a new node dataframe even if a file exists. Default is False.
        **kwargs : dict
            Additional arguments passed to the create_nodes() method.

        Returns:
        --------
        pandas.DataFrame or pyarrow.Table
            The loaded or newly created node data.

        Raises:
        -------
        ValueError
            If the 'name' field is missing from the created nodes dataframe.
        """

        if os.path.exists(self.filepath) and not from_scratch:
            logger.info(f"Trying to load {self.node_type} nodes from {self.filepath}")
            df = self.load_dataframe(filepath=self.filepath, columns=columns, include_cols=include_cols, **kwargs)
            return df
        
        logger.info(f"No node file found. Attempting to create {self.node_type} nodes")
        df = self.create_nodes(**kwargs)  # Subclasses will define this

        # Ensure the 'name' field is present
        if 'name' not in df.columns:
            raise ValueError(f"The 'name' field must be defined for {self.node_type} nodes. Define this in the create_nodes.")
        df['name'] = df['name']  # Ensure 'name' is set
        df['type'] = self.node_type  # Ensure 'type' is set
        if columns:
            df = df[columns]

        if not self.schema:
            logger.error(f"No schema set for {self.node_type} nodes")
            return None

        self.save_dataframe(df, self.filepath)
        return df
    
    def get_property_names(self):
        """
        Retrieves and logs the column names (properties) of the node data from the Parquet file.

        Returns:
        --------
        list of str
            A list of column names in the node file.
        """
        properties = Nodes.get_column_names(self.filepath)
        for property in properties:
            logger.info(f"Property: {property}")
        return properties

    def create_nodes(self, **kwargs):
        """
        Abstract method for creating nodes. Must be implemented by subclasses to define the logic 
        for creating nodes specific to the node type.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in a subclass.
        """
        if self.__class__.__name__ != 'Nodes':
            raise NotImplementedError("Subclasses must implement this method.")
        else:
            pass
    
    def create_schema(self, **kwargs):
        """
        Abstract method for creating a Parquet schema. Must be implemented by subclasses to define 
        the schema for the node data.

        Raises:
        -------
        NotImplementedError
            If this method is not implemented in a subclass.
        """
        if self.__class__.__name__ != 'Nodes':
            raise NotImplementedError("Subclasses must implement this method.")
        else:
            pass

    def load_dataframe(self, filepath, columns=None, include_cols=True, **kwargs):
        """
        Loads node data from a Parquet file, optionally filtering by columns.

        Parameters:
        -----------
        filepath : str
            Path to the Parquet file.
        columns : list of str, optional
            A list of column names to load. If None, all columns will be loaded.
        include_cols : bool, optional
            If True (default), the specified columns will be included. If False, 
            they will be excluded from the loaded dataframe.
        **kwargs : dict
            Additional arguments for reading the Parquet file.

        Returns:
        --------
        pandas.DataFrame or pyarrow.Table
            The loaded node data.
        """
        if not include_cols:
            metadata = pq.read_metadata(filepath)
            all_columns = []
            for filed_schema in metadata.schema:
                
                # Only want top column names
                max_defintion_level=filed_schema.max_definition_level
                if max_defintion_level!=1:
                    continue

                all_columns.append(filed_schema.name)

            columns = [col for col in all_columns if col not in columns]

        try:
            if self.output_format=='pandas':
                df = pd.read_parquet(filepath, columns=columns)
            elif self.output_format=='pyarrow':
                df = pq.read_table(filepath, columns=columns)

            return df
        except Exception as e:
            logger.error(f"Error loading {self.node_type} nodes from {filepath}: {e}")
            return None

    def save_dataframe(self, df, filepath):
        """
        Saves the given dataframe to a Parquet file at the specified filepath.

        Parameters:
        -----------
        df : pandas.DataFrame or pyarrow.Table
            The node data to save.
        filepath : str
            The path where the Parquet file should be saved.

        Raises:
        -------
        Exception
            If there is an error during the save process.
        """
        try:
            parquet_table = pa.Table.from_pandas(df, self.schema)
            pq.write_table(parquet_table, filepath)
            logger.info(f"Finished saving {self.node_type} nodes to {filepath}")
        except Exception as e:
            logger.error(f"Error converting dataframe to parquet table for saving: {e}")

    def to_neo4j(self, save_dir):
        """
        Converts the node data to a CSV file for importing into Neo4j. Saves the file in the given directory.

        Parameters:
        -----------
        save_dir : str
            Directory where the CSV file will be saved.
        """
        logger.info(f"Converting node to Neo4j : {self.filepath}")
        node_type=os.path.basename(self.filepath).split('.')[0]

        logger.debug(f"Node type: {node_type}")

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

        neo4j_column_name_mapping['type']=':LABEL'

        df=self.load_nodes(filepath=self.filepath)
        df.rename(columns=neo4j_column_name_mapping, inplace=True)
        df.index.name = f'{node_type}:ID({node_type}-ID)'

        os.makedirs(save_dir,exist_ok=True)

        save_file=os.path.join(save_dir,f'{node_type}.csv')
        logger.info(f"Saving {node_type} nodes to {save_file}")


        df.to_csv(save_file, index=True)

        logger.info(f"Finished converting node to Neo4j : {node_type}")
    
    @staticmethod
    def get_column_names(filepath):
        """
        Extracts and returns the top-level column names from a Parquet file.

        This method reads the metadata of a Parquet file and extracts the names of the top-level columns.
        It filters out nested columns or columns with a `max_definition_level` other than 1, ensuring that
        only primary, non-nested columns are included in the output.

        Args:
            filepath (str): The file path to the Parquet (.parquet) file.

        Returns:
            list of str: A list containing the names of the top-level columns in the Parquet file.

        Example:
            columns = Nodes.get_column_names('data/example.parquet')
            print(columns)
            # Output: ['column1', 'column2', 'column3']
        
        """
        metadata = pq.read_metadata(filepath)
        all_columns = []
        for filed_schema in metadata.schema:
            
            # Only want top column names
            max_defintion_level=filed_schema.max_definition_level
            if max_defintion_level!=1:
                continue

            all_columns.append(filed_schema.name)
        return all_columns
