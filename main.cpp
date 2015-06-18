#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Map.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <vector>
#include <map>
#include <array>

typedef Tpetra::DefaultPlatform::DefaultPlatformType Platform;
typedef Tpetra::Map<>::node_type node_type;
typedef Tpetra::Map<> map_type;
typedef Tpetra::CrsMatrix<> crs_matrix_type;
typedef Tpetra::CrsMatrix<>::global_ordinal_type GO;
typedef Tpetra::CrsMatrix<>::local_ordinal_type LO;
typedef Tpetra::CrsMatrix<>::scalar_type ST;
typedef Tpetra::global_size_t GST;
using Teuchos::Array;
using Teuchos::ArrayRCP;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::tuple;
using std::cerr;
using std::cout;
using std::endl;

void matrixAssembly( GO ielement, std::vector< std::vector<int> >  &nodes, std::vector<int> &indices, std::vector<int> &degrees,float matrix[4][4],RCP<crs_matrix_type> &sparseMatrix){
	/*
		Algorithm try to insert the element (matrix_i,matrix_j) from local matrix 
		to (sparse_i,sparse_j) in the sparse matrix
	*/
	std::vector<int> list_nodes = nodes[ielement]; //get the local nodes of ielement
	int node_i,node_j,matrix_i,matrix_j,sparse_i,sparse_j;
	// degree accumulators are variables for understand the position of a node in the F vector
	int degree_acc_i,degree_acc_j;

	int size_nodes = list_nodes.size(); //get number of nodes
	degree_acc_i = 0;
	for(int i = 0; i < size_nodes ; ++i){
		node_i = list_nodes[i]; //local nodes of an element correspond to a node index in a global node list
		for(int g1 = 0; g1 < degrees[node_i]; ++g1){ //for every variable in node
			matrix_i = degree_acc_i + g1;
			sparse_i = indices[node_i] + g1;
			degree_acc_j = 0;
			for(int j = 0; j < size_nodes; ++j){
				node_j = list_nodes[j];
				for(int g2 = 0; g2 < degrees[node_j]; ++g2){
					matrix_j = degree_acc_j + g2;
					sparse_j = indices[node_j] + g2;
					sparseMatrix->insertGlobalValues(  static_cast<GO>(sparse_i),
														tuple<GO> (static_cast<GO>(sparse_j)),
														tuple<ST> (matrix[matrix_i][matrix_j]));
				}
				degree_acc_j += degrees[node_j];
			}
		}
		degree_acc_i += degrees[node_i];
	}
}

int main(int argc, char *argv[] ){

	Teuchos::oblackholestream blackhole;
	Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);

	Platform &platform = Tpetra::DefaultPlatform::getDefaultPlatform();
	RCP<const Teuchos::Comm<int> > comm = platform.getComm();
	RCP<node_type> node = platform.getNode();
	const int myRank = comm->getRank();

	if(myRank == 0){
		cout << "Proc. size : " << comm->getSize() << endl; 
		//cout << "My rank : "  << myRank << endl;
	}

	//cout << "My rank : "  << myRank << endl;
	Teuchos::oblackholestream blackHole;
	std::ostream& out = (myRank == 0) ? std::cout : blackHole;

	/* TEST */
	GO indexBase = 0;
	GST numGblElements = 3;
	GST numGblEntries = 8;

	//int nodes_list[3][2] = {{1,2},{2,3},{3,4}}; 
	std::vector< std::vector < int >  > nodes;
	nodes.push_back(std::vector<int>());
	nodes.push_back(std::vector<int>());
	nodes.push_back(std::vector<int>());
	nodes.push_back(std::vector<int>());

	nodes[0].push_back(0);
	nodes[0].push_back(1);
	nodes[1].push_back(1);
	nodes[1].push_back(2);
	nodes[2].push_back(2);
	nodes[2].push_back(3);

	int degrees_list[4] = {2,2,2,2};
	std::vector<int> degrees(degrees_list,degrees_list+4);
	std::vector<int> indices;
	int aux = 0;    
    for (int k = 0; k < 4 ; ++k){
        indices.push_back(aux);
        aux += degrees[k];
    }

	RCP<const map_type> elementMap = rcp ( new map_type (numGblElements,indexBase,comm));
	RCP<const map_type> sparseMap = rcp ( new  map_type (numGblEntries,indexBase,comm));
	RCP<crs_matrix_type> sparseMatrix ( new crs_matrix_type (sparseMap,0));

	float matrices[3][4][4] = {{{252000,252000,-252000,252000},{252000,336000,-252000,168000},{-252000,-252000,252000,-252000},{252000,168000,-252000,336000}},{{252000,252000,-252000,252000},{252000,336000,-252000,168000},{-252000,-252000,252000,-252000},{252000,168000,-252000,336000}},{{126000,126000,-126000,126000},{126000,168000,-126000,84000},{-126000,-126000,126000,-126000},{126000,84000,-126000,168000}}};

	const size_t numMyElements = elementMap->getNodeNumElements ();
	for ( LO k = 0; k < static_cast<LO>(numMyElements); k++){
		GO element = elementMap->getGlobalElement(k);
		matrixAssembly(element,nodes,indices,degrees,matrices[element],sparseMatrix);
	}
	sparseMatrix->fillComplete();
	//cout << "My rank : " << myRank << " MyElements : " << numMyElements << endl;
	if(! sparseMatrix->isFillActive()){
		Tpetra::MatrixMarket::Writer<crs_matrix_type>::writeSparse(out,sparseMatrix);		
	}
}

