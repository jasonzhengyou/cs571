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

import static org.junit.Assert.assertEquals;

import java.io.FileInputStream;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import edu.emory.mathcs.nlp.common.util.Joiner;
import edu.emory.mathcs.nlp.component.util.feature.Field;
import edu.emory.mathcs.nlp.component.util.feature.Direction;
import edu.emory.mathcs.nlp.component.util.node.FeatMap;
import edu.emory.mathcs.nlp.component.util.reader.TSVReader;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class DEPNodeTest
{
	@Test
	public void test() throws Exception
	{
		TSVReader<DEPNode> reader = new TSVReader<>(new DEPIndex(1, 2, 3, 4, 5, 6));
		reader.open(new FileInputStream("src/main/resources/dat/wsj_0001.dep"));
		DEPNode[] nodes = reader.next();
		// TODO:
		System.out.println(Joiner.join(nodes, "\n"));
		
		//test dependent nodes
		assertEquals(nodes[0].getLeftMostDependent(), null); //since index 0 is at the left most position
		assertEquals(nodes[17].getRightMostDependent(), null); //since index 17 is at the right most position
		assertEquals(nodes[1].getLeftMostDependent(), nodes[0]);
		assertEquals(nodes[1].getRightMostDependent(), nodes[6]);
		//test siblings
		assertEquals(nodes[0].getLeftNearestSibling(), null); //since index 0 is at the left most position
		assertEquals(nodes[17].getRightNearestSibling(), null); //since index 17 is at the right most position
		assertEquals(nodes[5].getLeftNearestSibling(), nodes[2]); 
		assertEquals(nodes[5].getRightNearestSibling(), nodes[6]); 
		//test ansestors
		assertEquals(nodes[8].getAncestorSet().size(), 1); //since only has the root as an ancestor
		Set<DEPNode> ancestors = nodes[3].getAncestorSet();
		assertEquals(ancestors.size(), 5); 
		assertEquals(ancestors.contains(nodes[4]), true);
		assertEquals(ancestors.contains(nodes[5]), true);
		assertEquals(ancestors.contains(nodes[1]), true);
		assertEquals(ancestors.contains(nodes[8]), true);
		assertEquals(nodes[0].getLowestCommonAncestor(nodes[8]), nodes[8]); //lowest common ancestor should be the root when trying to get the common ancestor of node and root
		//test subNodes Set
		Set<DEPNode> subNodesSet = nodes[0].getSubNodeSet();
		assertEquals(subNodesSet.size(), 1); //since only node in sub tree should be itself
		subNodesSet = nodes[1].getSubNodeSet();
		for(int i = 1; i < 7; i++) {
			if (i == 1) continue;
			assertEquals(subNodesSet.contains(nodes[i]), true);
		}
		//test subNodeList
		List<DEPNode> subNodesList = nodes[0].getSubNodeList();
		assertEquals(subNodesList.size(), 1); //since only node in sub tree should be itself
		subNodesList = nodes[1].getSubNodeList();
		for(int i = 1; i < 7; i++) {
			if (i == 1) continue;
			assertEquals(subNodesList.contains(nodes[i]), true);
		}
		//test path
		Field field = Field.word_form;
		String path = nodes[10].getPath(nodes[10], Field.word_form); // the path to itself should just be itself
		assertEquals(path, nodes[10].getValue(field));
		path = nodes[4].getPath(nodes[8], Field.word_form);
		assertEquals(path, "^years^old^Vinken^join");		
		//test valency
		assertEquals(nodes[8].getValency(Direction.left), "<<");
		assertEquals(nodes[8].getValency(Direction.right), ">>");
		assertEquals(nodes[8].getValency(Direction.all), "<<->>");
		assertEquals(nodes[0].getValency(Direction.left), "0");
		assertEquals(nodes[0].getValency(Direction.right), "0");
		assertEquals(nodes[0].getValency(Direction.all), "0");


	}
	
	
	
	@Test
	public void testBasicFields()
	{
		DEPNode node = new DEPNode(1, "Jinho");
		
		assertEquals(1      , node.getID());
		assertEquals("Jinho", node.getWordForm());
		
		node = new DEPNode(1, "Jinho", "jinho", "NNP", new FeatMap("fst=jinho|lst=choi"));
		
		assertEquals(1       , node.getID());
		assertEquals("Jinho" , node.getWordForm());
		assertEquals("jinho" , node.getLemma());
		assertEquals("NNP"   , node.getPOSTag());
		
		node.removeFeat("fst");
		assertEquals(null  , node.getFeat("fst"));
		assertEquals("choi", node.getFeat("lst"));
		
		node.putFeat("fst", "Jinho");
		assertEquals("Jinho", node.getFeat("fst"));
	}
	
	@Test
	public void testSetters()
	{
		DEPNode node1 = new DEPNode(1, "He");
		DEPNode node2 = new DEPNode(2, "bought");
		DEPNode node3 = new DEPNode(3, "a");
		DEPNode node4 = new DEPNode(4, "car");
		
		node2.addDependent(node4, "dobj");
		node2.addDependent(node1, "nsubj");
		node4.addDependent(node3, "det");
		
		List<DEPNode> list = node2.getDependentList();
		assertEquals(node1, list.get(0));
		assertEquals(node4, list.get(1));
	}
}
