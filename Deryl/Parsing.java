import java.util.*;
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Parsing {

	public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
//			StringTokenizer itr = new StringTokenizer(value.toString());
//			while (itr.hasMoreTokens()) {
//				word.set(itr.nextToken());
//				context.write(word, one);
//			}
			
			String[] input = value.toString().split(",");
			
			if(input.length == 18)
			{
				String k =input[0];
				String v ="";
				int i = 0;
				boolean isAllPresent = true;
				for( i = 2; i< input.length -1; i++)
				{
					if(input[i] != null && !input[i].isEmpty())
					{
						v+=input[i]+",";
					}
					else
					{
						isAllPresent = false;
						break;
					}
				}
				if(isAllPresent)
				{
					if(input[i] != null && !input[i].isEmpty())
					{
						v+=input[i];
					}
					else
					{
						isAllPresent = false;
					
					}
					if(isAllPresent)
						context.write(new Text(k), new Text(v));
				}
				
			}
			
			
			
			
			
		}
	}

	public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
		

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			int[] average = new int[16];
			int counter =0;
			for (Text val : values) {
				String[] array = val.toString().split(",");
				for(int i=0; i< array.length; i++)
				{
					average[i] += Integer.parseInt(array[i]);
				}
				counter++;
			}
			String output="";
			int j=0;
			int result=0; 
			for(j=0; j < average.length-1; j++)
			{
				if(j == 0)
				{
//					output+= average[j]/counter +",";
					
//					output+=average[j]/counter+",";
					int a = average[j]/counter;
					if(33 <= a && a <= 65)
						result = 0;
					else if(66 <= a && a <= 70)
						result = 1;
					else if(71 <= a && a <= 80)
						result = 2;
					else if(81 <= a && a < 95)
						result = 3;
				}
				else
				{
					output+=(average[j]/counter)+",";
				}
			}
			output+=(average[j]/counter);
			output+=","+result;
			
			context.write(key, new Text(output));
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "Parsing");
		job.setJarByClass(Parsing.class);
		job.setMapperClass(TokenizerMapper.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
	//	job.setNumReduceTasks(0);
    	job.setReducerClass(IntSumReducer.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}
