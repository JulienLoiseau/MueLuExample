#include <iostream>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "MueLu.hpp"
#include "MueLu_TpetraOperator.hpp"
#include "MueLu_CreateTpetraPreconditioner.hpp"

using CrsMatrix_type = Tpetra::CrsMatrix<double, int, int>;
typedef Teuchos::Comm<int> Comm_type;
using Vector = Tpetra::Vector<double,int,int>;
using moperator = MueLu::TpetraOperator<double,int,int>; 
using toperator = Tpetra::Operator<double,int,int>;

int main(int argc, char* argv[]){

  MPI_Init(&argc,&argv); 

  Tpetra::MatrixMarket::Reader<CrsMatrix_type> rd;
  Teuchos::RCP<const Comm_type> comm = Teuchos::DefaultComm<int>::getComm();
  const Teuchos::RCP<CrsMatrix_type> A = rd.readSparseFile("matrix_a.mm",comm); 
  Vector x = Vector(A->getDomainMap());
  Vector y = Vector(A->getRangeMap());
  Vector r = Vector(y, Teuchos::Copy);

  y.putScalar(0); 
  x.randomize();
  A->apply(x,r); // y = A * x 
  // A x = y (y = 0)
  // r = y - A * x = - A * x
  r.update(-1 , r , 0); // r = -r

  Vector x_itr = Vector(x, Teuchos::Copy);
  Vector x_err = Vector(x, Teuchos::Copy);
  Vector dx = Vector(x, Teuchos::Copy);
  Vector Adx = Vector(y, Teuchos::Copy);
    
  x_itr.putScalar(0);
  dx.putScalar(0);
  std::string xml_plist = "preconditioner_muelu_plist.xml";
  Teuchos::RCP<toperator> op_A = A; 
  Teuchos::RCP<moperator> pc = MueLu::CreateTpetraPreconditioner(op_A,xml_plist); 

  for (int i=0; i!=10; ++i) {
    std::cout<<"### iteration: "<<i<<std::endl;
    pc->apply(r, dx); //             pc (A) :  A dx_0 = r => A dx_1 = r 
    x_itr.update(1, dx, 1); //                 x_itr = dx + x_itr
    x_err.update(-1, x_itr, 0); //             x_err = - x_itr 
    std::cout<<"dx: "<<dx.norm2()<<std::endl;
    std::cout<<"x_error: "<<x_err.norm2()<<std::endl;

    A->apply(x_itr, Adx);                  //     Adx = A * dx 
    r.update(-1, Adx, 0);               //     r = y - A * dx
  }


  MPI_Barrier(MPI_COMM_WORLD); 
  MPI_Finalize(); 

  return 0; 
} 
