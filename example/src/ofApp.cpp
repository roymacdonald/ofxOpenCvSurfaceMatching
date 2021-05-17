#include "ofApp.h"


//--------------------------------------------------------------
void ofApp::setup(){

	ofSetLogLevel(OF_LOG_VERBOSE);
	
	SM.train("parasaurolophus_low_normals2.ply");
	SM.match("rs1_normals.ply");

}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

//string modelFileName = ofToDataPath("parasaurolophus_low_normals2.ply", true);
//									//parasaurolophus_6700.ply
//string sceneFileName = ofToDataPath("rs1_normals.ply", true);
//
//Mat pc = loadPLYSimple(modelFileName.c_str(), 1);
//
//	// Now train the model
//	cout << "Training..." << endl;
//	int64 tick1 = cv::getTickCount();
//	ppf_match_3d::PPF3DDetector detector(0.025, 0.05);
//	detector.trainModel(pc);
//	int64 tick2 = cv::getTickCount();
//	cout << endl << "Training complete in "
//		 << (double)(tick2-tick1)/ cv::getTickFrequency()
//		 << " sec" << endl << "Loading model..." << endl;
//
//	// Read the scene
//	Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);
//
//	// Match the model to the scene and get the pose
//	cout << endl << "Starting matching..." << endl;
//	vector<Pose3DPtr> results;
//	tick1 = cv::getTickCount();
//	detector.match(pcTest, results, 1.0/40.0, 0.05);
//	tick2 = cv::getTickCount();
//	cout << endl << "PPF Elapsed Time " <<
//		 (tick2-tick1)/cv::getTickFrequency() << " sec" << endl;
//
//	//check results size from match call above
//	size_t results_size = results.size();
//	cout << "Number of matching poses: " << results_size;
//	if (results_size == 0) {
//		cout << endl << "No matching poses found. Exiting." << endl;
//		return;
//	}
//
//	// Get only first N results - but adjust to results size if num of results are less than that specified by N
//	size_t N = 2;
//	if (results_size < N) {
//		cout << endl << "Reducing matching poses to be reported (as specified in code): "
//			 << N << " to the number of matches found: " << results_size << endl;
//		N = results_size;
//	}
//	vector<Pose3DPtr> resultsSub(results.begin(),results.begin()+N);
//
//	// Create an instance of ICP
//	ICP icp(100, 0.005f, 2.5f, 8);
//	int64 t1 = cv::getTickCount();
//
//	// Register for all selected poses
//	cout << endl << "Performing ICP on " << N << " poses..." << endl;
//	icp.registerModelToScene(pc, pcTest, resultsSub);
//	int64 t2 = cv::getTickCount();
//
//	cout << endl << "ICP Elapsed Time " <<
//		 (t2-t1)/cv::getTickFrequency() << " sec" << endl;
//
//	cout << "Poses: " << endl;
//	// debug first five poses
//	for (size_t i=0; i<resultsSub.size(); i++)
//	{
//		Pose3DPtr result = resultsSub[i];
//		cout << "Pose Result " << i << endl;
//		result->printPose();
//		if (i==0)
//		{
//			Mat pct = transformPCPose(pc, result->pose);
//			writePLY(pct, "para6700PCTrans.ply");
//		}
//	}
//
//	return 0;
