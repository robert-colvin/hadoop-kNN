package edu.ucr.cs.cs226.rcolv001;

import java.io.IOException;
import java.util.StringTokenizer;
import java.lang.Math;
import java.util.PriorityQueue;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


/**
 * Calculate kNN
 *
 */

class KNNPoint 
{
    private String id;
    private double distance;

    public KNNPoint(String id, double distance) {
        this.id = id;
        this.distance = distance;
    }
    public double getDistance() {
        return this.distance;
    }
    public String getID(){
        return this.id;
    }
}
public class KNN 
{
    public static class CalcDistancesMapper extends Mapper<Object, Text, Text, DoubleWritable>{

        private Text pointID = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            double queryX = conf.getDouble("queryX", 0.0);
            double queryY = conf.getDouble("queryY", 0.0);
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                //each line is a comma separated line of the form "ID,xCoord,yCoord\n"
                String[] line = itr.nextToken("\n").split(",");
                pointID.set(line[0]);
                //get coords of current point and calculate euclidean distance
                double x = Double.parseDouble(line[1]);
                double y = Double.parseDouble(line[2]);
                double distance = getEuclideanDistance(x, y, queryX, queryY);

                //map each pointID to its distance from the query point
                context.write(pointID, new DoubleWritable(distance));
            }
        }

        private double getEuclideanDistance(double x1, double y1, double x2, double y2) {
            return Math.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));
        }
    }


    public static class KNearestReducer extends Reducer<Text,DoubleWritable,Text,DoubleWritable> {

        private DoubleWritable result = new DoubleWritable();
        
        //store as maxheap to exclude values
        private static PriorityQueue<KNNPoint> sortedPoints = new PriorityQueue<KNNPoint>(new Comparator<KNNPoint>() {
            public int compare(KNNPoint p1, KNNPoint p2) {
                double d = p1.getDistance() - p2.getDistance();
                if (d < 0)
                    return 1;
                if (d > 0)
                    return -1;

                return 0;
            }
        }); 
        
        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            double distance = 0;
            int k = context.getConfiguration().getInt("k", 0);
            for (DoubleWritable val : values) { //this loop better only iterate once per key
                distance += val.get();
            }
            
            if (sortedPoints.size() < k)
                sortedPoints.add(new KNNPoint(key.toString(), distance));
            else if (sortedPoints.size() >= k && distance < sortedPoints.peek().getDistance()) {
                sortedPoints.poll();
                sortedPoints.add(new KNNPoint(key.toString(), distance));
            }
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            int k = conf.getInt("k", 1);
            double queryX = conf.getDouble("queryX", 0.0);
            double queryY = conf.getDouble("queryY", 0.0);
            context.write(new Text("The " + k + " nearest points to query point (" + queryX + ", " + queryY + ") are:"), null);
            PriorityQueue<KNNPoint> results = new PriorityQueue<KNNPoint>(new Comparator<KNNPoint>() {
                public int compare(KNNPoint p1, KNNPoint p2) {
                    double d = p1.getDistance() - p2.getDistance();
                    if (d < 0)
                        return -1;
                    if (d > 0)
                        return 1;

                    return 0;
                }
            }); 
            for (int i = 0; i < k; i++) {
                KNNPoint curr = sortedPoints.poll();
                results.add(curr);
            }
             
            for (int i = 0; i < k; i++) {
                KNNPoint curr = results.poll();
                context.write(new Text(curr.getID()), new DoubleWritable(curr.getDistance()));
            }


        }
    }

    //takes 5 input arguments:
    //args[0] = path to csv file of points
    //args[1] = x coordinate of query point
    //args[2] = y coordinate of query point
    //args[3] = number of nearest neighbors to return (k)
    //args[4] (optional) = path to store program output; defaults to rcolv001KNNOutput if left blank
    public static void main(String[] args) throws Exception {
        String pointsPath = args[0];
        double queryPointX = Double.parseDouble(args[1]); //takes x and y coordinates as space separated floats
        double queryPointY = Double.parseDouble(args[2]);
        int k = Integer.parseInt(args[3]);
        String outputPath = "rcolv001KNNOutput";
        if (args.length >= 5)
            outputPath = args[4];
/*
        String pointsPath = "points";
        double queryPointX = 51.821;
        double queryPointY = 31.943;
        int k = 40; 
        String outputPath = "assn2output/";
*/
        Configuration conf = new Configuration();
        conf.set("pathToPoints", pointsPath);
        conf.setDouble("queryX", queryPointX);
        conf.setDouble("queryY", queryPointY);
        conf.setInt("k", k);

        Job job = Job.getInstance(conf, "knn");
        job.setJarByClass(KNN.class);
        job.setMapperClass(CalcDistancesMapper.class);
        job.setReducerClass(KNearestReducer.class);
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(job, new Path(pointsPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
