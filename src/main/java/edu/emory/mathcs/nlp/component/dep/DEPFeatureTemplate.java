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
package edu.emory.mathcs.nlp.component.dep;

import java.util.Arrays;

import edu.emory.mathcs.nlp.component.util.feature.Direction;
import edu.emory.mathcs.nlp.component.util.feature.FeatureItem;
import edu.emory.mathcs.nlp.component.util.feature.FeatureTemplate;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public abstract class DEPFeatureTemplate extends FeatureTemplate<DEPNode,DEPState<DEPNode>>
{
	private static final long serialVersionUID = -2218894375050796569L;
	private static final int MIN_LENGTH = 4;

	public DEPFeatureTemplate()	
	{
		init();
	}
	
	protected abstract void init();
	
//	========================= FEATURE EXTRACTORS =========================
	
	@Override
	protected String getFeature(FeatureItem<?> item)
	{
		DEPNode node = getNode(item);
		if (node == null) return null;
		int length;
		switch (item.field)
		{
		case word_form: return node.getWordForm();
		case simplified_word_form: return node.getSimplifiedWordForm();
		case uncapitalized_simplified_word_form: return node.getWordForm().toLowerCase();
		case lemma: return node.getLemma();
		case pos_tag: return node.getPOSTag();
		case feats: return node.getFeat((String)item.value);
		case dependency_label: return node.getLabel();
		case valency: return node.getValency((Direction)item.value);
		case suffix: 
			length = node.getSimplifiedWordForm().length();
			if (length < MIN_LENGTH) {
				return null;
			}
			return node.getSimplifiedWordForm().substring(length - 3, length);
		case prefix: 
			length = node.getSimplifiedWordForm().length();
			if (length < MIN_LENGTH) {
				return null;
			}
			return node.getSimplifiedWordForm().substring(0, 3);
		case capitalized:
			if  (!node.getWordForm().equals(node.getWordForm().toLowerCase())) {
				return "1";
			}
			return "0";

		default: throw new IllegalArgumentException("Unsupported feature: "+item.field);
		}
	}
	
	@Override
	protected String[] getFeatures(FeatureItem<?> item)
	{
		DEPNode node = getNode(item);
		if (node == null) return null;
		
		switch (item.field)
		{
		case binary: return getBinaryFeatures(node);
		default: throw new IllegalArgumentException("Unsupported feature: "+item.field);
		}
	}
	
	protected String[] getBinaryFeatures(DEPNode node)
	{
		String[] values = new String[2];
		int index = 0;
		
		if (state.isFirst(node)) values[index++] = "0";
		if (state.isLast (node)) values[index++] = "1";
		
		return (index == 0) ? null : (index == values.length) ? values : Arrays.copyOf(values, index);
	}
	
	protected DEPNode getNode(FeatureItem<?> item)
	{
		DEPNode node = null;
		
		switch (item.source)
		{
		case i: node = state.getStack (item.window); break;
		case j: node = state.getInput (item.window); break;
		case k: node = state.peekStack(item.window); break;
		}
		
		return getNode(node, item);
	}
	
	protected DEPNode getNode(DEPNode node, FeatureItem<?> item)
	{
		if (node == null || item.relation == null)
			return node;
		
		switch (item.relation)
		{
		case h   : return node.getHead();
		case h2  : return node.getGrandHead();
		case lmd : return node.getLeftMostDependent();
		case lmd2: return node.getLeftMostDependent(1);
		case lnd : return node.getLeftNearestDependent();
		case lnd2: return node.getLeftNearestDependent(1);
		case lns : return node.getLeftNearestSibling();
		case lns2: return node.getLeftNearestSibling(1);
		case rmd : return node.getRightMostDependent();
		case rmd2: return node.getRightMostDependent(1);
		case rnd : return node.getRightNearestDependent();
		case rnd2: return node.getRightNearestDependent(1);
		case rns : return node.getRightNearestSibling();
		case rns2: return node.getRightNearestSibling(1);
		}
		
		return null;
	}
}
