/*
* Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
* Released under terms of the BSD Simplified license.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*   * Neither the name of the organization nor the names of its contributors
*     may be used to endorse or promote products derived from this software
*     without specific prior written permission.
*
*   See <http://www.opensource.org/licenses/bsd-license>
*/


#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include "opencv2/core/core.hpp"

#include <vector>
#include <set>

//using namespace cv;
//using namespace std;

// Removes duplicate elements in a given vector.
template<typename _Tp>
inline std::vector<_Tp> remove_dups(const std::vector<_Tp>& src) {
  typedef typename set<_Tp>::const_iterator constSetIterator;
  typedef typename std::vector<_Tp>::const_iterator constVecIterator;
  set<_Tp> set_elems;
  for (std::constVecIterator it = src.begin(); it != src.end(); ++it)
    set_elems.insert(*it);
  std::vector<_Tp> elems;
  for (std::constSetIterator it = set_elems.begin(); it != set_elems.end(); ++it)
    elems.push_back(*it);
  return elems;
}

// The namespace cv provides opencv related helper functions.
namespace libfacerec {

  // Checks if a given matrix is symmetric, with an epsilon for floating point
  // matrices (1E-16 by default).
  //
  //      Mat mSymmetric = (Mat_<double>(2,2) << 1, 2, 2, 1);
  //      Mat mNonSymmetric = (Mat_<double>(2,2) << 1, 2, 3, 4);
  //      bool symmetric = isSymmetric(mSymmetric); // true
  //      bool not_symmetric = isSymmetric(mNonSymmetric); // false
  //
  bool isSymmetric(cv::InputArray src, double eps = 1E-16);

  // Sorts a 1D Matrix by given sort order and returns the sorted indices.
  // This is just a wrapper to simplify cv::sortIdx:
  //
  //      Mat mNotSorted = (Mat_<double>(1,4) << 1.0, 0.0, 3.0, -1.0);
  //      // to sort the vector use
  //      Mat sorted_indices = cv::argsort(mNotSorted, true);
  //      // make a conversion to vector<int>
  //      vector<int> sorted_indices = cv::argsort(mNotSorted, true);
  //
  cv::Mat argsort(cv::InputArray src, bool ascending = true);

  // Calculates a histogram for a given integral matrix. The minimum inclusive
  // value (minVal) and maximum inclusive value can be specified (optionally normed).
  cv::Mat histc(cv::InputArray src, int minVal = 0, int maxVal = 255, bool normed = false);

  // Reads a sequence from a FileNode::SEQ with type _Tp into a result vector.
  template<typename _Tp>
  inline void readFileNodeList(const cv::FileNode& fn, std::vector<_Tp>& result) {
	  if (fn.type() == cv::FileNode::SEQ) {
		  for (cv::FileNodeIterator it = fn.begin(); it != fn.end();) {
        _Tp item;
        it >> item;
        result.push_back(item);
      }
    }
  }

  // Writes the a list of given items to a cv::FileStorage.
  template<typename _Tp>
  inline void writeFileNodeList(cv::FileStorage& fs, const std::string& name,
	  const std::vector<_Tp>& items) {
    // typedefs
	  typedef typename std::vector<_Tp>::const_iterator constVecIterator;
    // write the elements in item to fs
    fs << name << "[";
	for (std::constVecIterator it = items.begin(); it != items.end(); ++it) {
      fs << *it;
    }
    fs << "]";
  }


  // Sorts a given matrix src by column for given indices.
  //
  // Note: create is called on dst.
  //void sortMatrixColumnsByIndices(cv::InputArray src, cv::InputArray indices, cv::OutputArray dst);

  // Sorts a given matrix src by row for given indices.
  //cv::Mat sortMatrixColumnsByIndices(cv::InputArray src, cv::InputArray indices);

  // Sorts a given matrix src by row for given indices.
  //
  // Note: create is called on dst.
  //void sortMatrixRowsByIndices(cv::InputArray src, cv::InputArray indices, cv::OutputArray dst);

  // Sorts a given matrix src by row for given indices.
  //cv::Mat sortMatrixRowsByIndices(cv::InputArray src, cv::InputArray indices);

  // Turns a vector of matrices into a row matrix.
  //cv::Mat asRowMatrix(cv::InputArrayOfArrays src, int rtype, double alpha = 1, double beta = 0);

  // Turns a vector of matrices into a column matrix.
  cv::Mat asColumnMatrix(cv::InputArrayOfArrays src, int rtype, double alpha = 1, double beta = 0);
  
} // //namespace cv

#endif