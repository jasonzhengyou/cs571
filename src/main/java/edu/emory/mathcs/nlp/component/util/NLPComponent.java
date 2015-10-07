/**
 * Copyright 2015, Emory University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.emory.mathcs.nlp.component.util;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;

import edu.emory.mathcs.nlp.common.collection.tuple.Pair;
import edu.emory.mathcs.nlp.component.dep.DEPState;
import edu.emory.mathcs.nlp.component.dep.DEPStatePrediction;
import edu.emory.mathcs.nlp.component.util.eval.Eval;
import edu.emory.mathcs.nlp.component.util.feature.FeatureTemplate;
import edu.emory.mathcs.nlp.component.util.state.NLPState;
import edu.emory.mathcs.nlp.learn.model.StringModel;
import edu.emory.mathcs.nlp.learn.util.Prediction;
import edu.emory.mathcs.nlp.learn.util.StringPrediction;
import edu.emory.mathcs.nlp.learn.vector.StringVector;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public abstract class NLPComponent<N,S extends NLPState<N>> implements Serializable
{
	private static final long serialVersionUID = 4546728532759275929L;
	protected FeatureTemplate<N,S> feature_template;
	protected StringModel[] models;
	protected NLPFlag flag;
	protected Eval eval;
	private static final double SCORE_THRES = .5;
	
//	============================== CONSTRUCTORS ==============================
	
	public NLPComponent(StringModel[] models)
	{
		setModels(models);
	}
	
//	============================== SERIALIZATION ==============================
	
	@SuppressWarnings("unchecked")
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException
	{
		feature_template = (FeatureTemplate<N,S>)in.readObject();
		models = (StringModel[])in.readObject();
		readLexicons(in);
	}
	
	private void writeObject(ObjectOutputStream out) throws IOException
	{
		out.writeObject(feature_template);
		out.writeObject(models);
		writeLexicons(out);
	}
	
	protected abstract void readLexicons (ObjectInputStream in) throws IOException, ClassNotFoundException;
	protected abstract void writeLexicons(ObjectOutputStream out) throws IOException;
	
//	============================== MODELS ==============================
	
	public StringModel[] getModels()
	{
		return models;
	}
	
	public void setModels(StringModel[] model)
	{
		this.models = model;
	}
	
//	============================== FEATURE ==============================

	public FeatureTemplate<N,S> getFeatureTemplate()
	{
		return feature_template;
	}

	public void setFeatureTemplate(FeatureTemplate<N,S> template)
	{
		feature_template = template;
	}
	
//	============================== FLAG ==============================
	
	public NLPFlag getFlag()
	{
		return flag;
	}
	
	public void setFlag(NLPFlag flag)
	{
		this.flag = flag;
	}
	
	public boolean isTrain()
	{
		return flag == NLPFlag.TRAIN;
	}
	
	public boolean isAggregate()
	{
		return flag == NLPFlag.AGGREGATE;
	}
	
//	public boolean isValidate()
//	{
//		return flag == NLPFlag.VALIDATE;
//	}
	
	public boolean isDecode()
	{
		return flag == NLPFlag.DECODE;
	}
	
	public boolean isEvaluate()
	{
		return flag == NLPFlag.EVALUATE;
	}
	
//	============================== EVALUATOR ==============================

	public Eval getEval()
	{
		return eval;
	}
	
	public void setEval(Eval eval)
	{
		this.eval = eval;
	}
	
//	============================== PROCESS ==============================
	
	/** @return the processing state for the input nodes. */
	protected abstract S createState(N[] nodes);
	/** @return the prediction made by the statistical model(s). */
	protected abstract StringPrediction getModelPrediction(S state, StringVector vector);
	/** Adds a training instance (label, x) to the statistical model. */
	protected abstract void addInstance(String label, StringVector vector);
	
	public void process(N[] nodes)
	{
		if ((!isDecode()) || isTrain() || isAggregate()) {
			processTrain(nodes);
			return;
		}
		
		S state = createState(nodes);
		boolean first = true;
		DEPStatePrediction prevPrediction = null;
		feature_template.setState(state);
		ArrayList<DEPStatePrediction> branchingStates = new ArrayList<DEPStatePrediction>();
		
		
		while (!state.isTerminate())
		{
			StringVector vector = extractFeatures(state);
			Pair<Prediction, Prediction> predictionPair = getPredictionBranching(state, vector);
			
			state.addToScore(predictionPair.o1.getScore());
			
			if (predictionPair.o1.getScore() < SCORE_THRES && !first && !isTrain()) {
				branchingStates.add(prevPrediction);
			}
			prevPrediction = new DEPStatePrediction((S) new DEPState((DEPState) state), predictionPair.o2);
			first = false;
			StringPrediction label = getPrediction(state, vector); //you need to change this
			state.next(label);
		}
		
		if (branchingStates.size() > (state.getSentenceLength()/2)){
			double maxScore = state.getScore();
			for(DEPStatePrediction branchState: branchingStates) {
				S branch = (S) branchState.getState();
				Prediction prediction = branchState.getPrediction();
				
				//get label from prediction
				StringPrediction label = getLabelFromPrediciton(prediction); 
				
				//pass to aux AND aux returns score
				branch.next(label);
				S finalState = processAux(branch);
				double branchScore = finalState.getScore();
				
				//compare score to max AND set max if necessary AND sets currentState if set max
				
				if(branchScore > maxScore) {
					maxScore = branchScore;
					state = finalState;
				}
			}
		}
				
		if (isEvaluate()) state.evaluate(eval);
	}
	public S processAux(S state)
	{
		feature_template.setState(state);
		
		while (!state.isTerminate())
		{
			StringVector vector = extractFeatures(state);
			StringPrediction label = getPrediction(state, vector);
			state.next(label);
		}
		return state;
	}
	
	public void processTrain(N[] nodes)
	{
		S state = createState(nodes);
		feature_template.setState(state);
		if (!isDecode()) state.saveOracle();
		
		while (!state.isTerminate())
		{
			StringVector vector = extractFeatures(state);
			if (isTrain() || isAggregate()) addInstance(state.getOraclePrediction(), vector);
			StringPrediction label = getPrediction(state, vector);
			state.next(label);
		}
	
		if (isEvaluate()) state.evaluate(eval);
	}
	
	/** @return the oracle prediction for training; otherwise, the model predict. */
	protected StringPrediction getPrediction(S state, StringVector vector)
	{
		return isTrain() ? new StringPrediction(state.getOraclePrediction(), 1) : getModelPrediction(state, vector);
	}
	
	protected Pair<Prediction,Prediction> getPredictionBranching(S state, StringVector vector)
	{
		return getModelPredictionBranching(state, vector);
		
	}
	/** @return the vector consisting of all features extracted from the state. */
	protected StringVector extractFeatures(S state)
	{
		return feature_template.extractFeatures();
	}

	protected Pair<Prediction,Prediction> getModelPredictionBranching(S state, StringVector vector) {
		// TODO Auto-generated method stub
		return null;
	}

	protected StringPrediction getLabelFromPrediciton(Prediction prediction) {
		// TODO Auto-generated method stub
		return null;
	}
}
