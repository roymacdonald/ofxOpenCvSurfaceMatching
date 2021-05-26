#include "ofApp.h"


//--------------------------------------------------------------
void ofApp::setup(){
	
	ofSetLogLevel(OF_LOG_VERBOSE);
	
	
	modelPath = "parasaurolophus_low_normals2.ply";
	scenePath = "rs1_normals.ply";
	
	modelMesh.load(modelPath);
//	modelMesh.setMode(OF_PRIMITIVE_POINTS);
//	modelMesh.setColorForIndices(0,  modelMesh.getNumIndices(), ofColor(255,0,0));
	
	
	sceneMesh.load(scenePath);
//	sceneMesh.setMode(OF_PRIMITIVE_POINTS);
//	sceneMesh.setColorForIndices(0,  sceneMesh.getNumIndices(), ofColor(0,0,255));
	
	
	
	easyCam.setFarClip(99999999);
	easyCam.lookAt(modelMesh.getCentroid());
	easyCam.setDistance(2000);
	
	
	light.setPosition(0, 0, 2000);
	light.lookAt(sceneMesh.getCentroid());
	
	
	/// register to trainingEndEvent so once trainAsync finishes, ofApp::onTrainEnd gets excecuted.
	trainEndListener = SM.trainingEndEvent.newListener(this, &ofApp::onTrainEnd);
		
	
	
	/// train the surface matcher with the model you want to find on the scene.
	/// with the provided models it takes a few minutes to train, be patient.
	SM.trainAsync(modelPath);
	
}

//--------------------------------------------------------------
void ofApp::onTrainEnd(){
	SM.match(scenePath);
	bUseMatrix = true;
	trainEndListener.unsubscribe();
}

//--------------------------------------------------------------
void ofApp::update(){
	
}

//--------------------------------------------------------------
void ofApp::draw(){
	
	ofEnableDepthTest();
	ofPushStyle();
	
    ofEnableLighting();
	light.enable();
	easyCam.begin();
	
	
			glPointSize(3);
	
	
	if(bShowScene) sceneMesh.draw();
	
	if(bShowModel){
		if(bUseMatrix){
			
			if(SM.beginApplyingPose(easyCam, matrixIndex)){
				modelMesh.draw();
				SM.endApplyingPose();
			}
		}else{
			modelMesh.draw();
		}
	}
	
	
	easyCam.end();
	light.disable();
	
    ofDisableLighting();
	ofPopStyle();
	ofDisableDepthTest();
	
	ofSetColor(255);
	stringstream ss;
	
	
	ss << "SM. num poses : " << SM.getNumPoses() << endl;
	ss << "SM is training: " << boolalpha << SM.isTraining() << endl;
	ss << "\nPress [ key ] for:\n";
	ss << " [1] bShowScene: " << boolalpha << bShowScene << endl;
	ss << " [2] bShowModel: " << boolalpha << bShowModel << endl;
	ss << " [m] bUseMatrix: " << boolalpha << bUseMatrix << endl;
	ss << " [n] matrixIndex: " << matrixIndex << endl;
	
	ofSetColor(255);
	
	ofDrawBitmapString(ss.str(), 20,20);
	
	
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	if(key == '1') bShowScene = !bShowScene;
	if(key == '2') bShowModel = !bShowModel;
	if(key == 'm') {
		bUseMatrix = !bUseMatrix;
	}
	if(key == 'n') {
		if(SM.getNumPoses() > 0){
			matrixIndex++;
			matrixIndex %= SM.getNumPoses();
		}
	}
	
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
